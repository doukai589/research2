#!/usr/bin/env python3
"""ST-EEGFormer-small + PhysioNetMI SAS-Cert workbench runner.

This is a workbench runner. Reusable pieces should be promoted into `sas_core`
after the protocol is validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "local_python_deps"
if str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

from sas_core.backbones.steegformer import build_steegformer
from sas_core.data.physionet_mi import build_physionet_mi_cache, default_physionet_mi_protocol, split_indices, support_test_split_for_subject
from sas_core.data.transforms import to_steegformer_input
from sas_core.metrics.classification import classification_metrics
from sas_core.utils.io import write_json
from sas_core.utils.seed import set_seed


OUT_DIR = Path(__file__).resolve().parent / "outputs"
CANONICAL_CACHE = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "data" / "physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz"
SFREQ = 160.0
CURRENT_GROUPS = ["NaiveAug_LS010", "ArtifactReject_LS010", "SoftWeight_noReject_LS010", "SASCert_SoftAR_LS010"]
COMPONENT_GATED_GROUPS = [
    "NaiveAug_LS010",
    "SoftWeight_noReject_LS010",
    "SASCert_SoftAR_LS010",
    "ArtifactGatePhysio_LS010",
    "ComponentGatedV1_LS010",
]


@dataclass
class Candidate:
    x: np.ndarray
    y: int
    original_local_index: int
    aug_type: str
    aug_id: str


def ranknorm(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks.astype(np.float32)


def clean_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    y = x.copy()
    signal_std = float(y.std() + 1e-6)
    y += rng.normal(0.0, signal_std * (0.025 + 0.03 * intensity), size=y.shape)
    max_shift = max(1, int(y.shape[1] * 0.015 * (1.0 + intensity)))
    y = np.roll(y, int(rng.integers(-max_shift, max_shift + 1)), axis=1)
    if rng.random() < 0.35:
        n_drop = max(1, int(round(y.shape[0] * 0.04)))
        chans = rng.choice(y.shape[0], size=n_drop, replace=False)
        y[chans] = y[chans].mean(axis=1, keepdims=True)
    return y.astype(np.float32)


def content_drift(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.55) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spectrum = np.fft.rfft(x, axis=1)
    mi_band = (freqs >= 8.0) & (freqs <= 30.0)
    band_idx = np.where(mi_band)[0]
    if len(band_idx):
        n_mask = max(1, int(round(len(band_idx) * np.clip(intensity, 0.0, 0.9))))
        chosen = rng.choice(band_idx, size=n_mask, replace=False)
        spectrum[:, chosen] *= 1.0 - 0.95 * intensity
        spectrum[:, chosen] *= np.exp(1j * rng.uniform(-np.pi, np.pi, size=(x.shape[0], n_mask)))
    return np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(np.float32)


def bad_artifact(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.55) -> np.ndarray:
    y = x.copy()
    t = np.arange(y.shape[1], dtype=np.float32) / SFREQ
    signal_std = float(y.std() + 1e-6)
    drift = np.sin(2 * np.pi * rng.uniform(0.2, 1.5) * t + rng.uniform(0, 2 * np.pi))
    y[: min(4, y.shape[0])] += signal_std * (0.7 + 1.8 * intensity) * drift
    burst_len = max(16, int(y.shape[1] * (0.06 + 0.08 * intensity)))
    start = int(rng.integers(0, max(1, y.shape[1] - burst_len)))
    chans = rng.choice(y.shape[0], size=max(1, int(2 + 4 * intensity)), replace=False)
    y[chans, start : start + burst_len] += rng.normal(0.0, signal_std * (1.0 + 2.5 * intensity), size=(len(chans), burst_len))
    return y.astype(np.float32)


def style_mismatch(x: np.ndarray, rng: np.random.Generator, style_bank: np.ndarray, intensity: float = 0.55) -> np.ndarray:
    ref = style_bank[int(rng.integers(0, len(style_bank)))]
    x_mean, x_std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
    r_mean, r_std = ref.mean(axis=1, keepdims=True), ref.std(axis=1, keepdims=True) + 1e-6
    styled = (x - x_mean) / x_std * r_std + r_mean
    return ((1 - intensity) * x + intensity * styled).astype(np.float32)


def bad_physio(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.55) -> np.ndarray:
    y = x.copy()
    n_perm = max(2, int(round(y.shape[0] * intensity)))
    chosen = rng.choice(y.shape[0], size=n_perm, replace=False)
    shuffled = chosen.copy()
    rng.shuffle(shuffled)
    y[chosen] = y[shuffled]
    return y.astype(np.float32)


def generate_candidates(
    support_x: np.ndarray,
    support_y: np.ndarray,
    style_bank: np.ndarray,
    rng: np.random.Generator,
    per_trial: int = 5,
) -> List[Candidate]:
    aug_types = ["clean", "clean", "bad_artifact", "bad_content", "bad_physio"]
    candidates: list[Candidate] = []
    for i, (x, label) in enumerate(zip(support_x, support_y)):
        for k in range(per_trial):
            aug_type = aug_types[k % len(aug_types)]
            if aug_type == "clean":
                aug_x = clean_aug(x, rng)
            elif aug_type == "bad_artifact":
                aug_x = bad_artifact(x, rng)
            elif aug_type == "bad_content":
                aug_x = content_drift(x, rng)
            elif aug_type == "bad_physio":
                aug_x = bad_physio(x, rng)
            else:
                aug_x = style_mismatch(x, rng, style_bank)
            candidates.append(Candidate(aug_x.astype(np.float32), int(label), i, aug_type, f"{aug_type}_{i:03d}_{k:02d}"))
    return candidates


def bandpower_features(x: np.ndarray) -> np.ndarray:
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    psd = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(psd[..., mask].mean(axis=-1))
    return np.stack(feats, axis=-1).astype(np.float32)


def covariance_flat(x: np.ndarray) -> np.ndarray:
    covs = []
    for sample in x:
        cov = np.cov(sample) + np.eye(sample.shape[0]) * 1e-5
        covs.append(cov.reshape(-1))
    return np.stack(covs).astype(np.float32)


def artifact_risk_features(x: np.ndarray) -> np.ndarray:
    diff = np.diff(x, axis=-1)
    low = np.abs(x.mean(axis=-1))
    high = diff.std(axis=-1)
    energy = (x**2).mean(axis=-1)
    kurt = np.nan_to_num(kurtosis(x, axis=-1, fisher=False), nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([low, high, energy, kurt], axis=1).astype(np.float32)


def l2_to_anchor(x: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.sqrt(((x - anchor) ** 2).mean(axis=1))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8)


def score_candidates(
    candidates: Sequence[Candidate],
    support_x: np.ndarray,
    support_features: np.ndarray,
    candidate_features: np.ndarray,
    artifact_gate_percentile: float,
) -> List[Dict[str, object]]:
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    orig_x = np.stack([support_x[c.original_local_index] for c in candidates]).astype(np.float32)
    orig_feat = np.stack([support_features[c.original_local_index] for c in candidates]).astype(np.float32)

    content_raw = cosine(candidate_features, orig_feat)

    cand_stats = np.concatenate([cand_x.mean(axis=2), cand_x.std(axis=2)], axis=1)
    support_stats = np.concatenate([support_x.mean(axis=2).mean(axis=0), support_x.std(axis=2).mean(axis=0)])[None, :]
    style_raw = -l2_to_anchor(cand_stats, support_stats)

    physio_raw = -(
        l2_to_anchor(bandpower_features(cand_x)[..., 2:4].reshape(len(cand_x), -1), bandpower_features(orig_x)[..., 2:4].reshape(len(orig_x), -1))
        + l2_to_anchor(covariance_flat(cand_x), covariance_flat(orig_x))
    )

    artifact_feats = artifact_risk_features(cand_x)
    support_art = artifact_risk_features(support_x)
    med = np.median(support_art, axis=0, keepdims=True)
    mad = np.median(np.abs(support_art - med), axis=0, keepdims=True) + 1e-6
    artifact_risk = np.maximum(0.0, (artifact_feats - med) / mad).mean(axis=1)

    content = ranknorm(content_raw, True)
    style = ranknorm(style_raw, True)
    physio = ranknorm(physio_raw, True)
    artifact_safe = ranknorm(artifact_risk, False)
    sas_score = 0.55 * artifact_safe + 0.45 * content
    artifact_threshold = float(np.percentile(artifact_risk, artifact_gate_percentile))
    artifact_gate_pass = artifact_risk < artifact_threshold
    score_artifact_gate_physio = physio.copy()
    score_artifact_gate_physio[~artifact_gate_pass] = 0.0
    component_base = ranknorm(0.75 * physio + 0.25 * style, True)
    component_gated_v1 = component_base * artifact_gate_pass.astype(np.float32)
    component_gated_v1_soft = component_base * (0.2 + 0.8 * artifact_safe)

    rows = []
    for i, cand in enumerate(candidates):
        rows.append(
            {
                "aug_id": cand.aug_id,
                "original_local_index": cand.original_local_index,
                "aug_type": cand.aug_type,
                "label": cand.y,
                "content_score": float(content[i]),
                "style_score": float(style[i]),
                "physio_score": float(physio[i]),
                "artifact_safe_score": float(artifact_safe[i]),
                "artifact_risk_raw": float(artifact_risk[i]),
                "sas_score": float(sas_score[i]),
                "artifact_gate_pass": int(artifact_gate_pass[i]),
                "score_artifact_gate_physio": float(score_artifact_gate_physio[i]),
                "component_gated_v1": float(component_gated_v1[i]),
                "component_gated_v1_soft": float(component_gated_v1_soft[i]),
            }
        )
    return rows


def extract_features(model: nn.Module, x160: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    x_st = to_steegformer_input(x160)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_st), batch_size):
            xb = torch.from_numpy(x_st[start : start + batch_size]).float().to(device)
            feats = model.extract_features(xb)
            out.append(feats.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "untagged"


def load_optional_steegformer_state(model: nn.Module, state_dict_path: str | None) -> Dict[str, object] | None:
    if not state_dict_path:
        return None
    path = Path(state_dict_path)
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
    target = model.model if hasattr(model, "model") else model
    target_state = target.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape.append(f"{key}: ckpt={tuple(value.shape)} model={tuple(target_state[key].shape)}")
            continue
        load_state[key] = value
    msg = target.load_state_dict(load_state, strict=False)
    return {
        "state_dict_path": str(path),
        "checkpoint_keys": len(state_dict),
        "target_model_keys": len(target_state),
        "loaded_keys": len(load_state),
        "skipped_missing_name": len(skipped_missing),
        "skipped_shape": skipped_shape[:80],
        "missing_after_load": list(msg.missing_keys)[:120],
        "unexpected_after_load": list(msg.unexpected_keys)[:120],
    }


class FeatureHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    loss = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    init_state: Dict[str, torch.Tensor] | None,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    label_smoothing: float,
) -> FeatureHead:
    head = FeatureHead(features.shape[1]).to(device)
    if init_state is not None:
        head.load_state_dict(init_state)
    ds = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).long(), torch.from_numpy(weights).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        head.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = weighted_ce(head(xb), yb, wb, label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite head loss")
            loss.backward()
            opt.step()
    return head


def predict_probs(head: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            xb = torch.from_numpy(features[start : start + 256]).float().to(device)
            out.append(torch.softmax(head(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def group_aug_selection(group: str, score_rows: Sequence[Dict[str, object]], artifact_reject_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    content = np.asarray([float(r["content_score"]) for r in score_rows], dtype=np.float32)
    artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in score_rows], dtype=np.float32)
    artifact_gate_physio = np.asarray([float(r.get("score_artifact_gate_physio", 0.0)) for r in score_rows], dtype=np.float32)
    component_gated_v1 = np.asarray([float(r.get("component_gated_v1", 0.0)) for r in score_rows], dtype=np.float32)
    keep = np.ones(len(score_rows), dtype=bool)
    weights = np.ones(len(score_rows), dtype=np.float32)
    threshold = np.percentile(artifact_risk, artifact_reject_percentile)
    if group in {"ArtifactReject_LS010", "SASCert_SoftAR_LS010"}:
        keep = artifact_risk < threshold
    if group == "SoftWeight_noReject_LS010":
        weights = 0.2 + 0.8 * ranknorm(content, True)
    elif group == "SASCert_SoftAR_LS010":
        weights = 0.2 + 0.8 * ranknorm(content, True)
    elif group == "ArtifactGatePhysio_LS010":
        weights = 0.2 + 0.8 * ranknorm(artifact_gate_physio, True)
    elif group == "ComponentGatedV1_LS010":
        weights = 0.2 + 0.8 * ranknorm(component_gated_v1, True)
    return keep, weights


def train_source_head_state(args: argparse.Namespace, data: Dict[str, np.ndarray], original_features: np.ndarray) -> Dict[str, torch.Tensor]:
    splits = split_indices(data["subjects"])
    source_idx = splits["train"]
    source_head = train_head(
        original_features[source_idx],
        data["y"][source_idx],
        np.ones(len(source_idx), dtype=np.float32),
        None,
        args.device,
        args.source_epochs,
        args.lr,
        args.batch_size,
        0.0,
    )
    return {k: v.detach().cpu().clone() for k, v in source_head.state_dict().items()}


def run_fold(
    args: argparse.Namespace,
    data: Dict[str, np.ndarray],
    original_features: np.ndarray,
    model: nn.Module,
    source_state: Dict[str, torch.Tensor],
    target: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    X, y, subjects = data["X"], data["y"], data["subjects"]
    splits = split_indices(subjects)
    support_idx, test_idx = support_test_split_for_subject(y, subjects, target, args.shot_per_class, seed)
    source_idx = splits["train"]

    support_x = X[support_idx]
    support_y = y[support_idx]
    support_features = original_features[support_idx]
    test_features = original_features[test_idx]
    test_y = y[test_idx]

    style_bank = X[source_idx]
    candidates = generate_candidates(support_x, support_y, style_bank, rng, per_trial=args.n_aug)
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    cand_features = extract_features(model, cand_x, args.device, args.feature_batch_size)
    score_rows = score_candidates(candidates, support_x, support_features, cand_features, args.artifact_reject_percentile)

    score_path = OUT_DIR / "score_rows" / args.output_tag / f"target{target}_seed{seed}.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    rows = []
    for group in args.groups:
        keep, aug_w = group_aug_selection(group, score_rows, args.artifact_reject_percentile)
        train_features = [support_features]
        train_y = [support_y]
        train_w = [np.ones(len(support_y), dtype=np.float32)]
        if keep.any():
            train_features.append(cand_features[keep])
            train_y.append(np.asarray([c.y for c in candidates], dtype=np.int64)[keep])
            train_w.append(aug_w[keep].astype(np.float32))
        feat = np.concatenate(train_features, axis=0)
        labels = np.concatenate(train_y, axis=0)
        weights = np.concatenate(train_w, axis=0)
        head = train_head(feat, labels, weights, source_state, args.device, args.finetune_epochs, args.lr, args.batch_size, 0.10)
        probs = predict_probs(head, test_features, args.device)
        metrics = classification_metrics(test_y, probs)
        rows.append(
            {
                "target_subject": target,
                "seed": seed,
                "group": group,
                "n_support": len(support_idx),
                "n_aug_total": len(candidates),
                "n_aug_kept": int(keep.sum()),
                **metrics,
            }
        )
    return rows


def load_or_extract_original_features(args: argparse.Namespace, data: Dict[str, np.ndarray], model: nn.Module) -> np.ndarray:
    feat_path = OUT_DIR / "features" / f"original_st_features_{args.feature_tag}.npz"
    if feat_path.exists() and not args.rebuild_features:
        return np.load(feat_path)["features"].astype(np.float32)
    features = extract_features(model, data["X"], args.device, args.feature_batch_size)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feat_path,
        features=features,
        subjects=data["subjects"],
        labels=data["y"],
        feature_tag=np.asarray(args.feature_tag),
        steegformer_state_dict=np.asarray(args.steegformer_state_dict or ""),
    )
    return features


def paired_comparisons(rows: Sequence[Dict[str, object]], primary_group: str, baseline_groups: Sequence[str]) -> List[Dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["target_subject"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["group"])] = dict(row)
    pairs = []
    for (target, seed), group_rows in grouped.items():
        primary = group_rows.get(primary_group)
        if primary is None:
            continue
        for baseline in baseline_groups:
            base = group_rows.get(baseline)
            if base is None:
                continue
            pairs.append(
                {
                    "target_subject": target,
                    "seed": seed,
                    "comparison": f"{primary_group}-vs-{baseline}",
                    "delta_balanced_accuracy": float(primary["balanced_accuracy"]) - float(base["balanced_accuracy"]),
                    "delta_macro_f1": float(primary["macro_f1"]) - float(base["macro_f1"]),
                    "delta_ece": float(primary["ece"]) - float(base["ece"]),
                    "delta_nll": float(primary["nll"]) - float(base["nll"]),
                    "delta_brier": float(primary["brier"]) - float(base["brier"]),
                    "win_macro_f1": int(float(primary["macro_f1"]) > float(base["macro_f1"])),
                }
            )
    return pairs


def summarize(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    smoke: bool,
    experiment: str,
    groups: Sequence[str],
    primary_group: str,
    baseline_groups: Sequence[str],
) -> Dict[str, object]:
    by_group: dict[str, dict[str, float]] = {}
    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        by_group[group] = {
            key: float(np.mean([float(r[key]) for r in group_rows]))
            for key in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]
        }
    primary = by_group[primary_group]
    naive = by_group["NaiveAug_LS010"]
    comparison = {
        "primary_group": primary_group,
        "baseline_group": "NaiveAug_LS010",
        "delta_balanced_accuracy": primary["balanced_accuracy"] - naive["balanced_accuracy"],
        "delta_macro_f1": primary["macro_f1"] - naive["macro_f1"],
        "delta_ece": primary["ece"] - naive["ece"],
        "delta_nll": primary["nll"] - naive["nll"],
        "delta_brier": primary["brier"] - naive["brier"],
    }
    pairs = paired_comparisons(rows, primary_group, baseline_groups)
    naive_pairs = [p for p in pairs if p["comparison"].endswith("NaiveAug_LS010")]
    artifact_pairs = [p for p in pairs if p["comparison"].endswith("ArtifactReject_LS010")]
    soft_pairs = [p for p in pairs if p["comparison"].endswith("SoftWeight_noReject_LS010")]
    subject_wins = []
    for target in sorted(set(int(r["target_subject"]) for r in rows)):
        target_pairs = [p for p in naive_pairs if int(p["target_subject"]) == target]
        if target_pairs:
            subject_wins.append(float(np.mean([p["win_macro_f1"] for p in target_pairs])) > 0.5)
    seed_wins = []
    for seed in sorted(set(int(r["seed"]) for r in rows)):
        seed_pairs = [p for p in naive_pairs if int(p["seed"]) == seed]
        if seed_pairs:
            seed_wins.append(float(np.mean([p["win_macro_f1"] for p in seed_pairs])) > 0.5)
    comparison["subject_win_rate_macro_f1"] = float(np.mean(subject_wins)) if subject_wins else float("nan")
    comparison["seed_win_rate_macro_f1"] = float(np.mean(seed_wins)) if seed_wins else float("nan")
    payload = {
        "stage": "steegformer_sascert_core_workbench",
        "status": "completed_smoke" if smoke else "completed",
        "experiment": experiment,
        "primary_group": primary_group,
        "groups": by_group,
        "primary_vs_naive": comparison,
        "sascert_vs_naive": comparison if primary_group == "SASCert_SoftAR_LS010" else None,
        "paired_comparisons": {
            p["comparison"]: {
                key: float(np.mean([row[key] for row in pairs if row["comparison"] == p["comparison"]]))
                for key in ["delta_balanced_accuracy", "delta_macro_f1", "delta_ece", "delta_nll", "delta_brier", "win_macro_f1"]
            }
            for p in pairs
        },
        "component_ablation": {
            "beats_artifact_reject_macro_f1": bool(np.mean([p["delta_macro_f1"] for p in artifact_pairs]) > 0) if artifact_pairs else None,
            "beats_softweight_no_reject_macro_f1": bool(np.mean([p["delta_macro_f1"] for p in soft_pairs]) > 0) if soft_pairs else None,
            "mean_delta_macro_f1_vs_artifact_reject": float(np.mean([p["delta_macro_f1"] for p in artifact_pairs])) if artifact_pairs else None,
            "mean_delta_macro_f1_vs_softweight_no_reject": float(np.mean([p["delta_macro_f1"] for p in soft_pairs])) if soft_pairs else None,
        },
        "n_rows": len(rows),
    }
    write_json(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--targets", nargs="+", type=int, default=[90])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shot-per-class", type=int, default=5)
    parser.add_argument("--n-aug", type=int, default=5)
    parser.add_argument("--source-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--steegformer-state-dict", default=None, help="Optional source/validation-trained raw ST-EEGFormer state_dict.")
    parser.add_argument("--feature-tag", default=None, help="Feature cache tag. Defaults to pretrained or the state-dict file stem.")
    parser.add_argument("--output-tag", default=None, help="Output file tag. Defaults to feature-tag plus smoke/full.")
    parser.add_argument("--artifact-reject-percentile", type=float, default=90.0)
    parser.add_argument("--experiment", choices=["current", "component_gated"], default="current")
    args = parser.parse_args()
    if args.feature_tag is None:
        args.feature_tag = safe_tag(Path(args.steegformer_state_dict).stem if args.steegformer_state_dict else "pretrained")
    else:
        args.feature_tag = safe_tag(args.feature_tag)
    if args.output_tag is None:
        args.output_tag = safe_tag(f"{args.feature_tag}_{args.experiment}_{'smoke' if args.smoke else 'full'}")
    else:
        args.output_tag = safe_tag(args.output_tag)
    if args.experiment == "component_gated":
        args.groups = COMPONENT_GATED_GROUPS
        args.primary_group = "ComponentGatedV1_LS010"
        args.baseline_groups = ["NaiveAug_LS010", "SoftWeight_noReject_LS010", "SASCert_SoftAR_LS010", "ArtifactGatePhysio_LS010"]
    else:
        args.groups = CURRENT_GROUPS
        args.primary_group = "SASCert_SoftAR_LS010"
        args.baseline_groups = ["NaiveAug_LS010", "ArtifactReject_LS010", "SoftWeight_noReject_LS010"]
    return args


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(3407)
    if args.smoke:
        args.targets = args.targets[:1]
        args.seeds = args.seeds[:1]
        args.source_epochs = min(args.source_epochs, 2)
        args.finetune_epochs = min(args.finetune_epochs, 4)

    protocol = default_physionet_mi_protocol()
    data = build_physionet_mi_cache(CANONICAL_CACHE, protocol=protocol, rebuild=False)
    model, audit = build_steegformer(n_channels=data["X"].shape[1])
    tuned_audit = load_optional_steegformer_state(model, args.steegformer_state_dict)
    if tuned_audit:
        audit["optional_source_tuned_state_dict"] = tuned_audit
    audit["feature_tag"] = args.feature_tag
    audit["experiment"] = args.experiment
    audit["groups"] = args.groups
    audit["artifact_reject_percentile"] = args.artifact_reject_percentile
    model.to(args.device)
    write_json(OUT_DIR / "steegformer_load_audit.json", audit)
    original_features = load_or_extract_original_features(args, data, model)
    source_state = train_source_head_state(args, data, original_features)

    all_rows: list[dict[str, object]] = []
    for target in args.targets:
        for seed in args.seeds:
            print(f"running target={target} seed={seed}", flush=True)
            all_rows.extend(run_fold(args, data, original_features, model, source_state, target, seed))

    metrics_path = OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    pairs = paired_comparisons(all_rows, args.primary_group, args.baseline_groups)
    paired_path = OUT_DIR / f"steegformer_physionetmi_paired_comparison_{args.output_tag}.csv"
    if pairs:
        with paired_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)
    summary_path = OUT_DIR / f"compact_steegformer_sascert_result_{args.output_tag}.json"
    summary = summarize(all_rows, summary_path, args.smoke, args.experiment, args.groups, args.primary_group, args.baseline_groups)
    report = [
        "# ST-EEGFormer PhysioNetMI SAS-Cert Workbench Report",
        "",
        f"- Targets: `{args.targets}`",
        f"- Seeds: `{args.seeds}`",
        f"- Smoke: `{args.smoke}`",
        f"- Feature tag: `{args.feature_tag}`",
        f"- Output tag: `{args.output_tag}`",
        f"- Experiment: `{args.experiment}`",
        f"- Groups: `{args.groups}`",
        f"- Primary group: `{args.primary_group}`",
        f"- Artifact reject percentile: `{args.artifact_reject_percentile}`",
        "",
        "## Primary vs Naive",
        "",
    ]
    for key, value in summary["primary_vs_naive"].items():
        if isinstance(value, str):
            report.append(f"- `{key}`: `{value}`")
        else:
            report.append(f"- `{key}`: `{value:.6f}`")
    (OUT_DIR / f"STEEGFORMER_SASCERT_REPORT_{args.output_tag}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
