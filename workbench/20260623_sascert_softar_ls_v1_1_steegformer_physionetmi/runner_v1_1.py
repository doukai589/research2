#!/usr/bin/env python3
"""ST-EEGFormer-small + PhysioNetMI SAS-Cert-SoftAR-LS v1.1 runner.

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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis
from scipy.special import rel_entr
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "local_python_deps"
if str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))
REFERENCE_PY_TARGET = ROOT / "outputs" / "reference_algorithm_audit" / "python_target"
if str(REFERENCE_PY_TARGET) not in sys.path:
    sys.path.insert(0, str(REFERENCE_PY_TARGET))

try:
    from pyriemann.utils.distance import distance_riemann

    PYRIEMANN_STATUS = "used"
except Exception as exc:  # pragma: no cover - runtime audit path
    distance_riemann = None
    PYRIEMANN_STATUS = f"fallback:{exc}"

try:
    from mne_features.feature_extraction import extract_features as mne_extract_features

    MNE_FEATURES_STATUS = "used"
except Exception as exc:  # pragma: no cover - runtime audit path
    mne_extract_features = None
    MNE_FEATURES_STATUS = f"fallback:{exc}"

try:
    from autoreject import AutoReject  # noqa: F401

    AUTOREJECT_STATUS = "available_offline_not_integrated"
except Exception as exc:  # pragma: no cover - runtime audit path
    AUTOREJECT_STATUS = f"not_integrated:{exc}"

from sas_core.backbones.steegformer import build_steegformer
from sas_core.data.physionet_mi import build_physionet_mi_cache, default_physionet_mi_protocol, split_indices, support_test_split_for_subject
from sas_core.data.transforms import to_steegformer_input
from sas_core.metrics.classification import classification_metrics
from sas_core.utils.io import write_json
from sas_core.utils.seed import set_seed


OUT_DIR = Path(__file__).resolve().parent / "outputs"
CANONICAL_CACHE = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "data" / "physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz"
SFREQ = 160.0
V1_1_GROUPS = ["NaiveAug_LS010", "ArtifactReject_LS010", "SoftWeight_noReject_LS010", "SAS-Cert-SoftAR-LS-v1.1"]
V1_1_PRIMARY = "SAS-Cert-SoftAR-LS-v1.1"
V1_2_GROUPS = ["NaiveAug_LS010", "SoftWeight_noReject_LS010", "SAS-Cert-SoftSafe-LS-v1.2"]
V1_2_PRIMARY = "SAS-Cert-SoftSafe-LS-v1.2"
V1_1_FULL_OUT = ROOT / "workbench" / "20260623_sascert_softar_ls_v1_1_steegformer_physionetmi" / "outputs"
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation.*")
warnings.filterwarnings("ignore", message="No module named 'numba'. Your code will be slower.*")


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


def gaussian_noise_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    signal_std = float(x.std() + 1e-6)
    return (x + rng.normal(0.0, signal_std * (0.025 + 0.025 * intensity), size=x.shape)).astype(np.float32)


def time_shift_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    max_shift = max(1, int(x.shape[1] * (0.01 + 0.02 * intensity)))
    return np.roll(x, int(rng.integers(-max_shift, max_shift + 1)), axis=1).astype(np.float32)


def time_crop_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    crop_frac = 0.04 + 0.04 * intensity
    crop_len = max(8, int(x.shape[1] * crop_frac))
    start = int(rng.integers(0, max(1, x.shape[1] - crop_len)))
    y = x.copy()
    fill = 0.5 * (y[:, start : start + 1] + y[:, start + crop_len - 1 : start + crop_len])
    y[:, start : start + crop_len] = fill
    return y.astype(np.float32)


def frequency_mask_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spectrum = np.fft.rfft(x, axis=1)
    valid = np.where((freqs >= 6.0) & (freqs <= 45.0))[0]
    if len(valid):
        width = max(1, int(round(len(valid) * (0.04 + 0.06 * intensity))))
        center = int(rng.choice(valid))
        lo, hi = max(0, center - width), min(len(freqs), center + width + 1)
        spectrum[:, lo:hi] *= 0.15
    return np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(np.float32)


def channel_dropout_aug(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    y = x.copy()
    n_drop = max(1, int(round(y.shape[0] * (0.03 + 0.04 * intensity))))
    chans = rng.choice(y.shape[0], size=n_drop, replace=False)
    y[chans] = y[chans].mean(axis=1, keepdims=True)
    return y.astype(np.float32)


def mild_frequency_mixup_aug(x: np.ndarray, rng: np.random.Generator, style_bank: np.ndarray, intensity: float = 0.25) -> np.ndarray:
    ref = style_bank[int(rng.integers(0, len(style_bank)))]
    spec_x = np.fft.rfft(x, axis=1)
    spec_r = np.fft.rfft(ref, axis=1)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    mask = (freqs >= 8.0) & (freqs <= 30.0)
    alpha = 0.08 + 0.12 * intensity
    mixed = spec_x.copy()
    mixed[:, mask] = (1.0 - alpha) * spec_x[:, mask] + alpha * spec_r[:, mask]
    return np.fft.irfft(mixed, n=x.shape[1], axis=1).astype(np.float32)


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
    per_trial: int = 6,
) -> List[Candidate]:
    aug_types = ["gaussian_noise", "time_shift", "time_crop", "frequency_mask", "channel_dropout", "mild_frequency_mixup"]
    candidates: list[Candidate] = []
    for i, (x, label) in enumerate(zip(support_x, support_y)):
        for k in range(per_trial):
            aug_type = aug_types[k % len(aug_types)]
            if aug_type == "gaussian_noise":
                aug_x = gaussian_noise_aug(x, rng)
            elif aug_type == "time_shift":
                aug_x = time_shift_aug(x, rng)
            elif aug_type == "time_crop":
                aug_x = time_crop_aug(x, rng)
            elif aug_type == "frequency_mask":
                aug_x = frequency_mask_aug(x, rng)
            elif aug_type == "channel_dropout":
                aug_x = channel_dropout_aug(x, rng)
            else:
                aug_x = mild_frequency_mixup_aug(x, rng, style_bank)
            candidates.append(Candidate(aug_x.astype(np.float32), int(label), i, aug_type, f"{aug_type}_{i:03d}_{k:02d}"))
    return candidates


def bandpower_features(x: np.ndarray) -> np.ndarray:
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 79), (58, 62)]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    psd = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(psd[..., mask].mean(axis=-1) if mask.any() else np.zeros(psd.shape[:-1], dtype=np.float32))
    return np.stack(feats, axis=-1).astype(np.float32)


def covariance_matrices(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    covs = []
    for sample in x:
        cov = np.cov(sample).astype(np.float64)
        cov += np.eye(sample.shape[0], dtype=np.float64) * eps
        covs.append(cov)
    return np.stack(covs).astype(np.float64)


def covariance_flat(x: np.ndarray) -> np.ndarray:
    covs = []
    for cov in covariance_matrices(x):
        covs.append(cov.reshape(-1))
    return np.stack(covs).astype(np.float32)


def artifact_risk_features(x: np.ndarray) -> np.ndarray:
    diff = np.diff(x, axis=-1)
    bp = bandpower_features(x)
    low = bp[:, :, 0]
    high = bp[:, :, 5]
    line = bp[:, :, 6]
    energy = (x**2).mean(axis=-1)
    kurt = np.nan_to_num(kurtosis(x, axis=-1, fisher=False), nan=0.0, posinf=0.0, neginf=0.0)
    skewness = np.nan_to_num(((x - x.mean(axis=-1, keepdims=True)) ** 3).mean(axis=-1) / ((x.std(axis=-1) + 1e-6) ** 3), nan=0.0, posinf=0.0, neginf=0.0)
    fast_change = diff.std(axis=-1)
    return np.concatenate([low, high, line, energy, kurt, skewness, fast_change], axis=1).astype(np.float32)


def mne_feature_stats(x: np.ndarray) -> tuple[np.ndarray, str]:
    if mne_extract_features is None:
        stats = np.concatenate([x.mean(axis=2), x.std(axis=2)], axis=1)
        return stats.astype(np.float32), MNE_FEATURES_STATUS
    try:
        feats = mne_extract_features(x.astype(np.float64), SFREQ, ["mean", "std", "kurtosis", "skewness"], n_jobs=1, return_as_df=False)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), "used"
    except Exception as exc:
        stats = np.concatenate([x.mean(axis=2), x.std(axis=2)], axis=1)
        return stats.astype(np.float32), f"fallback:{exc}"


def pairwise_riemann_distance(cand_x: np.ndarray, orig_x: np.ndarray) -> tuple[np.ndarray, int, str]:
    cand_cov = covariance_matrices(cand_x)
    orig_cov = covariance_matrices(orig_x)
    distances: list[float] = []
    bad = 0
    status = PYRIEMANN_STATUS
    for a, b in zip(cand_cov, orig_cov):
        try:
            if distance_riemann is None:
                raise RuntimeError("pyRiemann unavailable")
            d = float(distance_riemann(a, b))
        except Exception as exc:
            status = f"fallback:{exc}"
            d = float(np.linalg.norm(a - b, ord="fro"))
        if not np.isfinite(d):
            bad += 1
            d = float(np.linalg.norm(a - b, ord="fro"))
        distances.append(d)
    return np.asarray(distances, dtype=np.float32), bad, status


def build_class_prototypes(source_features: np.ndarray, source_y: np.ndarray, support_features: np.ndarray, support_y: np.ndarray) -> dict[int, np.ndarray]:
    feats = np.concatenate([source_features, support_features], axis=0)
    labels = np.concatenate([source_y, support_y], axis=0)
    return {int(label): feats[labels == label].mean(axis=0).astype(np.float32) for label in sorted(set(labels.astype(int)))}


def prototype_margin(candidate_features: np.ndarray, labels: np.ndarray, prototypes: dict[int, np.ndarray]) -> np.ndarray:
    proto_labels = sorted(prototypes)
    proto_mat = np.stack([prototypes[k] for k in proto_labels]).astype(np.float32)
    denom = (np.linalg.norm(candidate_features, axis=1, keepdims=True) * np.linalg.norm(proto_mat, axis=1)[None, :]) + 1e-8
    sims = candidate_features @ proto_mat.T / denom
    out = []
    for i, label in enumerate(labels.astype(int)):
        y_pos = proto_labels.index(int(label))
        own = sims[i, y_pos]
        other = np.max(np.delete(sims[i], y_pos)) if len(proto_labels) > 1 else 0.0
        out.append(float(own - other))
    return np.asarray(out, dtype=np.float32)


def kl_consistency(orig_probs: np.ndarray, cand_probs: np.ndarray) -> np.ndarray:
    kl = np.sum(rel_entr(np.clip(orig_probs, 1e-7, 1.0), np.clip(cand_probs, 1e-7, 1.0)), axis=1)
    return (-kl).astype(np.float32)


def l2_to_anchor(x: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.sqrt(((x - anchor) ** 2).mean(axis=1))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8)


def score_candidates(
    candidates: Sequence[Candidate],
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_features: np.ndarray,
    source_features: np.ndarray,
    source_y: np.ndarray,
    candidate_features: np.ndarray,
    orig_probs: np.ndarray,
    cand_probs: np.ndarray,
    artifact_gate_percentile: float,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    orig_x = np.stack([support_x[c.original_local_index] for c in candidates]).astype(np.float32)
    orig_feat = np.stack([support_features[c.original_local_index] for c in candidates]).astype(np.float32)
    labels = np.asarray([c.y for c in candidates], dtype=np.int64)

    e_embed_raw = cosine(candidate_features, orig_feat)
    prototypes = build_class_prototypes(source_features, source_y, support_features, support_y)
    e_proto_raw = prototype_margin(candidate_features, labels, prototypes)
    e_pred_raw = kl_consistency(orig_probs, cand_probs)

    cand_mne, mne_status = mne_feature_stats(cand_x)
    support_mne, _ = mne_feature_stats(support_x)
    cand_bp = bandpower_features(cand_x).reshape(len(cand_x), -1)
    support_bp = bandpower_features(support_x).reshape(len(support_x), -1)
    cand_cov_summary = covariance_flat(cand_x)
    support_cov_summary = covariance_flat(support_x)
    cand_style = np.concatenate([cand_x.mean(axis=2), cand_x.std(axis=2), cand_mne, cand_bp, cand_cov_summary], axis=1)
    support_style_anchor = np.concatenate(
        [
            support_x.mean(axis=2).mean(axis=0),
            support_x.std(axis=2).mean(axis=0),
            support_mne.mean(axis=0),
            support_bp.mean(axis=0),
            support_cov_summary.mean(axis=0),
        ]
    )[None, :]
    style_distance = l2_to_anchor(cand_style, support_style_anchor)
    style_raw = -style_distance

    orig_bp = bandpower_features(orig_x).reshape(len(orig_x), -1)
    d_band = l2_to_anchor(cand_bp, orig_bp)
    d_cov, cov_bad, pyriemann_status = pairwise_riemann_distance(cand_x, orig_x)
    physio_distance = d_band + d_cov
    physio_raw = -physio_distance

    artifact_feats = artifact_risk_features(cand_x)
    support_art = artifact_risk_features(support_x)
    med = np.median(support_art, axis=0, keepdims=True)
    mad = np.median(np.abs(support_art - med), axis=0, keepdims=True) + 1e-6
    artifact_risk = np.maximum(0.0, (artifact_feats - med) / mad).mean(axis=1)

    e_embed = ranknorm(e_embed_raw, True)
    e_proto = ranknorm(e_proto_raw, True)
    e_pred = ranknorm(e_pred_raw, True)
    content = e_embed + e_proto + 0.5 * e_pred
    content_rank = ranknorm(content, True)
    style = ranknorm(style_raw, True)
    physio = ranknorm(physio_raw, True)
    artifact_safe = ranknorm(artifact_risk, False)
    artifact_threshold = float(np.percentile(artifact_risk, artifact_gate_percentile))
    artifact_gate_pass = artifact_risk < artifact_threshold
    e_total = content_rank + physio + 0.5 * style
    total_rank = ranknorm(e_total, True)
    soft_weight = 0.2 + 0.8 * total_rank
    softar_weight = soft_weight.copy()
    softar_weight[~artifact_gate_pass] = 0.0
    artifact_med = float(np.median(artifact_risk))
    artifact_mad = float(np.median(np.abs(artifact_risk - artifact_med)) + 1e-6)
    artifact_robust_z = (artifact_risk - artifact_med) / artifact_mad
    extreme_artifact_outlier = (~np.isfinite(artifact_risk)) | (artifact_robust_z > 4.0)
    content_v1_2 = ranknorm(e_embed_raw, True) + ranknorm(e_proto_raw, True)
    content_v1_2_rank = ranknorm(content_v1_2, True)
    e_base_v1_2 = content_v1_2 + physio + 0.3 * style
    e_total_v1_2 = e_base_v1_2 * (0.5 + 0.5 * artifact_safe)
    softsafe_weight_v1_2 = 0.5 + 0.5 * ranknorm(e_total_v1_2, True)
    softsafe_weight_v1_2[extreme_artifact_outlier] = 0.0

    rows = []
    for i, cand in enumerate(candidates):
        rows.append(
            {
                "aug_id": cand.aug_id,
                "original_local_index": cand.original_local_index,
                "aug_type": cand.aug_type,
                "label": cand.y,
                "e_embed_raw": float(e_embed_raw[i]),
                "e_proto_raw": float(e_proto_raw[i]),
                "e_pred_raw": float(e_pred_raw[i]),
                "content_score": float(content_rank[i]),
                "style_score": float(style[i]),
                "physio_score": float(physio[i]),
                "artifact_safe_score": float(artifact_safe[i]),
                "artifact_risk_raw": float(artifact_risk[i]),
                "d_band": float(d_band[i]),
                "d_cov_riemann": float(d_cov[i]),
                "d_style": float(style_distance[i]),
                "e_total": float(e_total[i]),
                "artifact_gate_pass": int(artifact_gate_pass[i]),
                "soft_weight_v1_1": float(soft_weight[i]),
                "softar_weight_v1_1": float(softar_weight[i]),
                "artifact_robust_z": float(artifact_robust_z[i]),
                "extreme_artifact_outlier": int(extreme_artifact_outlier[i]),
                "content_score_v1_2": float(content_v1_2_rank[i]),
                "e_base_v1_2": float(e_base_v1_2[i]),
                "e_total_v1_2": float(e_total_v1_2[i]),
                "softsafe_weight_v1_2": float(softsafe_weight_v1_2[i]),
            }
        )
    audit = {
        "artifact_threshold": artifact_threshold,
        "artifact_extreme_threshold_robust_z": 4.0,
        "rejected_ratio": float(1.0 - artifact_gate_pass.mean()),
        "extreme_rejected_ratio_v1_2": float(extreme_artifact_outlier.mean()),
        "mean_soft_weight": float(soft_weight.mean()),
        "mean_softar_weight": float(softar_weight.mean()),
        "mean_softsafe_weight_v1_2": float(softsafe_weight_v1_2.mean()),
        "pyriemann_status": pyriemann_status,
        "mne_features_status": mne_status,
        "covariance_nan_inf_count": int(cov_bad),
        "score_nan_inf_count": int(
            np.sum(~np.isfinite(e_total))
            + np.sum(~np.isfinite(artifact_risk))
            + np.sum(~np.isfinite(d_band))
            + np.sum(~np.isfinite(d_cov))
        ),
    }
    return rows, audit


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


def train_head_real_aug_normalized(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    aug_features: np.ndarray,
    aug_labels: np.ndarray,
    aug_weights: np.ndarray,
    init_state: Dict[str, torch.Tensor] | None,
    device: str,
    epochs: int,
    lr: float,
    label_smoothing: float,
) -> FeatureHead:
    head = FeatureHead(real_features.shape[1]).to(device)
    if init_state is not None:
        head.load_state_dict(init_state)
    real_x = torch.from_numpy(real_features).float().to(device)
    real_y = torch.from_numpy(real_labels).long().to(device)
    aug_x = torch.from_numpy(aug_features).float().to(device)
    aug_y = torch.from_numpy(aug_labels).long().to(device)
    aug_w = torch.from_numpy(aug_weights).float().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        real_loss = F.cross_entropy(head(real_x), real_y, label_smoothing=label_smoothing)
        aug_loss_each = F.cross_entropy(head(aug_x), aug_y, reduction="none", label_smoothing=label_smoothing)
        aug_loss = (aug_loss_each * aug_w).sum() / aug_w.sum().clamp_min(1e-6)
        loss = real_loss + aug_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite normalized head loss")
        loss.backward()
        opt.step()
    return head


def loss_mass_diagnostics(
    head: nn.Module,
    real_features: np.ndarray,
    real_labels: np.ndarray,
    aug_features: np.ndarray,
    aug_labels: np.ndarray,
    aug_weights: np.ndarray,
    device: str,
    label_smoothing: float,
) -> Dict[str, float]:
    head.eval()
    with torch.no_grad():
        real_x = torch.from_numpy(real_features).float().to(device)
        real_y = torch.from_numpy(real_labels).long().to(device)
        aug_x = torch.from_numpy(aug_features).float().to(device)
        aug_y = torch.from_numpy(aug_labels).long().to(device)
        aug_w = torch.from_numpy(aug_weights).float().to(device)
        ce_real = F.cross_entropy(head(real_x), real_y, label_smoothing=label_smoothing)
        ce_aug_each = F.cross_entropy(head(aug_x), aug_y, reduction="none", label_smoothing=label_smoothing)
        ce_aug_raw = ce_aug_each.mean()
        ce_aug_weighted = (ce_aug_each * aug_w).sum() / aug_w.sum().clamp_min(1e-6)
        final_loss = ce_real + ce_aug_weighted
    return {
        "CE_real": float(ce_real.detach().cpu()),
        "CE_aug_raw": float(ce_aug_raw.detach().cpu()),
        "CE_aug_weighted": float(ce_aug_weighted.detach().cpu()),
        "final_train_loss": float(final_loss.detach().cpu()),
    }


def predict_probs(head: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            xb = torch.from_numpy(features[start : start + 256]).float().to(device)
            out.append(torch.softmax(head(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def group_aug_selection(group: str, score_rows: Sequence[Dict[str, object]], artifact_reject_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in score_rows], dtype=np.float32)
    e_total = np.asarray([float(r["e_total"]) for r in score_rows], dtype=np.float32)
    soft_weight = np.asarray([float(r["soft_weight_v1_1"]) for r in score_rows], dtype=np.float32)
    softsafe_weight = np.asarray([float(r.get("softsafe_weight_v1_2", 1.0)) for r in score_rows], dtype=np.float32)
    extreme = np.asarray([int(float(r.get("extreme_artifact_outlier", 0))) == 1 for r in score_rows], dtype=bool)
    keep = np.ones(len(score_rows), dtype=bool)
    weights = np.ones(len(score_rows), dtype=np.float32)
    threshold = np.percentile(artifact_risk, artifact_reject_percentile)
    if group in {"ArtifactReject_LS010", V1_1_PRIMARY}:
        keep = artifact_risk < threshold
    if group == "SoftWeight_noReject_LS010":
        weights = soft_weight
    elif group == V1_1_PRIMARY:
        weights = 0.2 + 0.8 * ranknorm(e_total, True)
        weights[~keep] = 0.0
    elif group == V1_2_PRIMARY:
        keep = ~extreme
        weights = softsafe_weight
        weights[~keep] = 0.0
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
    source_features = original_features[source_idx]
    source_y = y[source_idx]

    style_bank = X[source_idx]
    candidates = generate_candidates(support_x, support_y, style_bank, rng, per_trial=args.n_aug)
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    cand_features = extract_features(model, cand_x, args.device, args.feature_batch_size)

    source_head = FeatureHead(original_features.shape[1]).to(args.device)
    source_head.load_state_dict(source_state)
    orig_for_candidates = np.stack([support_features[c.original_local_index] for c in candidates]).astype(np.float32)
    orig_probs = predict_probs(source_head, orig_for_candidates, args.device)
    cand_probs = predict_probs(source_head, cand_features, args.device)

    score_rows, fold_cert = score_candidates(
        candidates,
        support_x,
        support_y,
        support_features,
        source_features,
        source_y,
        cand_features,
        orig_probs,
        cand_probs,
        args.artifact_reject_percentile,
    )

    score_path = OUT_DIR / "score_rows" / args.output_tag / f"target{target}_seed{seed}.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    rows = []
    cand_labels = np.asarray([c.y for c in candidates], dtype=np.int64)
    for group in args.groups:
        keep, aug_w = group_aug_selection(group, score_rows, args.artifact_reject_percentile)
        if args.experiment == "v1_2":
            if not keep.any():
                raise RuntimeError("v1_2 has no augmented candidates after extreme artifact filtering")
            aug_features = cand_features[keep]
            aug_labels = cand_labels[keep]
            aug_weights = aug_w[keep].astype(np.float32)
            head = train_head_real_aug_normalized(
                support_features,
                support_y,
                aug_features,
                aug_labels,
                aug_weights,
                source_state,
                args.device,
                args.finetune_epochs,
                args.lr,
                0.10,
            )
            loss_diag = loss_mass_diagnostics(head, support_features, support_y, aug_features, aug_labels, aug_weights, args.device, 0.10)
            sum_weight = float(aug_weights.sum())
            effective_aug_loss_scale = float(sum_weight / max(1, len(candidates)))
        else:
            train_features = [support_features]
            train_y = [support_y]
            train_w = [np.ones(len(support_y), dtype=np.float32)]
            if keep.any():
                train_features.append(cand_features[keep])
                train_y.append(cand_labels[keep])
                train_w.append(aug_w[keep].astype(np.float32))
            feat = np.concatenate(train_features, axis=0)
            labels = np.concatenate(train_y, axis=0)
            weights = np.concatenate(train_w, axis=0)
            head = train_head(feat, labels, weights, source_state, args.device, args.finetune_epochs, args.lr, args.batch_size, 0.10)
            loss_diag = {"CE_real": float("nan"), "CE_aug_raw": float("nan"), "CE_aug_weighted": float("nan"), "final_train_loss": float("nan")}
            sum_weight = float(aug_w[keep].sum()) if keep.any() else 0.0
            effective_aug_loss_scale = float(sum_weight / max(1, len(candidates)))
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
                "rejected_ratio": float(1.0 - keep.mean()),
                "mean_aug_weight_all": float(aug_w.mean()),
                "mean_aug_weight_kept": float(aug_w[keep].mean()) if keep.any() else 0.0,
                "sum_aug_weight_kept": sum_weight,
                "sum_weight_per_candidate": effective_aug_loss_scale,
                **loss_diag,
                "artifact_threshold": float(fold_cert["artifact_threshold"]),
                "fold_score_nan_inf_count": int(fold_cert["score_nan_inf_count"]),
                "fold_covariance_nan_inf_count": int(fold_cert["covariance_nan_inf_count"]),
                **metrics,
            }
        )
    fold_summary_path = OUT_DIR / "fold_certificate_summaries" / args.output_tag / f"target{target}_seed{seed}.json"
    fold_summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        fold_summary_path,
        {
            "target_subject": target,
            "seed": seed,
            **fold_cert,
            "n_support": int(len(support_idx)),
            "n_target_test": int(len(test_idx)),
            "n_source_train": int(len(source_idx)),
            "n_aug_total": int(len(candidates)),
            "leakage_audit": {
                "artifact_threshold_from": "candidate_aug_pool_from_target_support_only",
                "ranknorm_from": "candidate_aug_pool_from_target_support_only",
                "prototype_from": "source_train_plus_target_support",
                "style_anchor_from": "target_support_only",
                "target_test_used_for_scoring": False,
                "target_test_used_for_training": False,
                "target_test_used_for_final_eval_only": True,
            },
        },
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
        "sascert_vs_naive": comparison if primary_group in {V1_1_PRIMARY, V1_2_PRIMARY} else None,
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


def read_score_rows(output_tag: str) -> List[Dict[str, object]]:
    rows: list[dict[str, object]] = []
    score_dir = OUT_DIR / "score_rows" / output_tag
    if not score_dir.exists():
        return rows
    for path in sorted(score_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["score_file"] = path.name
                rows.append(row)
    return rows


def read_score_rows_from_dir(score_dir: Path) -> List[Dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not score_dir.exists():
        return rows
    for path in sorted(score_dir.glob("*.csv")):
        m = re.search(r"target(\d+)_seed(\d+)", path.name)
        target = int(m.group(1)) if m else -1
        seed = int(m.group(2)) if m else -1
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["score_file"] = path.name
                row["target_subject"] = target
                row["seed"] = seed
                rows.append(row)
    return rows


def write_table_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_distribution_rows(score_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_cols = [
        "artifact_risk_raw",
        "artifact_safe_score",
        "content_score",
        "content_score_v1_2",
        "physio_score",
        "style_score",
        "e_total",
        "e_total_v1_2",
        "soft_weight_v1_1",
        "softar_weight_v1_1",
        "softsafe_weight_v1_2",
        "d_band",
        "d_cov_riemann",
        "d_style",
    ]
    out = []
    for aug_type in ["ALL"] + sorted(set(str(r["aug_type"]) for r in score_rows)):
        subset = score_rows if aug_type == "ALL" else [r for r in score_rows if str(r["aug_type"]) == aug_type]
        row: dict[str, object] = {"aug_type": aug_type, "n": len(subset)}
        for col in numeric_cols:
            vals = np.asarray([float(r[col]) for r in subset if col in r and str(r[col]) != ""], dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_std"] = float(np.std(vals)) if len(vals) else float("nan")
            row[f"{col}_p10"] = float(np.percentile(vals, 10)) if len(vals) else float("nan")
            row[f"{col}_p90"] = float(np.percentile(vals, 90)) if len(vals) else float("nan")
        out.append(row)
    return out


def write_v1_1_gate_harm_audit() -> Dict[str, float]:
    rows = read_score_rows_from_dir(V1_1_FULL_OUT / "score_rows" / "v1_1_source_tuned_full")
    audit_rows = []
    for row in rows:
        audit_rows.append(
            {
                "target_subject": row["target_subject"],
                "seed": row["seed"],
                "rejected_v1_1": int(float(row["artifact_gate_pass"])) == 0,
                "E_embed": row["e_embed_raw"],
                "E_proto": row["e_proto_raw"],
                "E_pred": row["e_pred_raw"],
                "E_content": row["content_score"],
                "D_band": row["d_band"],
                "D_cov": row["d_cov_riemann"],
                "E_physio": row["physio_score"],
                "D_style": row["d_style"],
                "E_style": row["style_score"],
                "artifact_score": row["artifact_risk_raw"],
                "label": row["label"],
                "subject": row["target_subject"],
                "augmentation_type": row["aug_type"],
                "raw_CE_loss": "",
            }
        )
    write_table_csv(OUT_DIR / "gate_harm_audit.csv", audit_rows)
    rejected = [r for r in rows if int(float(r["artifact_gate_pass"])) == 0]
    kept = [r for r in rows if int(float(r["artifact_gate_pass"])) == 1]
    def mean_col(subset: Sequence[Dict[str, object]], col: str) -> float:
        return float(np.mean([float(r[col]) for r in subset])) if subset else float("nan")
    return {
        "n": float(len(rows)),
        "rejected_ratio": float(len(rejected) / len(rows)) if rows else float("nan"),
        "rejected_content_mean": mean_col(rejected, "content_score"),
        "kept_content_mean": mean_col(kept, "content_score"),
        "rejected_e_embed_mean": mean_col(rejected, "e_embed_raw"),
        "kept_e_embed_mean": mean_col(kept, "e_embed_raw"),
        "rejected_e_proto_mean": mean_col(rejected, "e_proto_raw"),
        "kept_e_proto_mean": mean_col(kept, "e_proto_raw"),
        "rejected_physio_mean": mean_col(rejected, "physio_score"),
        "kept_physio_mean": mean_col(kept, "physio_score"),
    }


def prior_v1_1_loss_mass_rows() -> List[Dict[str, object]]:
    score_rows = read_score_rows_from_dir(V1_1_FULL_OUT / "score_rows" / "v1_1_source_tuned_full")
    by_fold: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in score_rows:
        by_fold.setdefault((int(row["target_subject"]), int(row["seed"])), []).append(row)
    methods = ["NaiveAug_LS010", "ArtifactReject_LS010", "SoftWeight_noReject_LS010", V1_1_PRIMARY]
    out = []
    for method in methods:
        fold_stats = []
        for rows in by_fold.values():
            n = len(rows)
            artifact_keep = np.asarray([int(float(r["artifact_gate_pass"])) == 1 for r in rows], dtype=bool)
            if method == "NaiveAug_LS010":
                weights = np.ones(n, dtype=np.float32)
                keep = np.ones(n, dtype=bool)
            elif method == "ArtifactReject_LS010":
                weights = np.ones(n, dtype=np.float32)
                keep = artifact_keep
            elif method == "SoftWeight_noReject_LS010":
                weights = np.asarray([float(r["soft_weight_v1_1"]) for r in rows], dtype=np.float32)
                keep = np.ones(n, dtype=bool)
            else:
                weights = np.asarray([float(r["softar_weight_v1_1"]) for r in rows], dtype=np.float32)
                keep = artifact_keep
            sum_w = float(weights[keep].sum())
            fold_stats.append((float(weights.mean()), sum_w, n, sum_w / max(1, n)))
        out.append(
            {
                "source": "v1_1_prior_score_rows",
                "method": method,
                "mean_weight": float(np.mean([s[0] for s in fold_stats])),
                "sum_weight": float(np.mean([s[1] for s in fold_stats])),
                "num_candidates": float(np.mean([s[2] for s in fold_stats])),
                "effective_aug_loss_scale": float(np.mean([s[3] for s in fold_stats])),
                "CE_real": "",
                "CE_aug_raw": "",
                "CE_aug_weighted": "",
                "final_train_loss": "",
            }
        )
    return out


def rejected_summary_rows(score_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for aug_type in ["ALL"] + sorted(set(str(r["aug_type"]) for r in score_rows)):
        subset = score_rows if aug_type == "ALL" else [r for r in score_rows if str(r["aug_type"]) == aug_type]
        rejected = [r for r in subset if int(float(r["artifact_gate_pass"])) == 0]
        out.append(
            {
                "aug_type": aug_type,
                "n": len(subset),
                "n_rejected": len(rejected),
                "rejected_ratio": float(len(rejected) / len(subset)) if subset else float("nan"),
                "artifact_risk_mean": float(np.mean([float(r["artifact_risk_raw"]) for r in subset])) if subset else float("nan"),
                "rejected_artifact_risk_mean": float(np.mean([float(r["artifact_risk_raw"]) for r in rejected])) if rejected else float("nan"),
            }
        )
    return out


def failure_case_rows(score_rows: Sequence[Dict[str, object]], limit: int = 40) -> List[Dict[str, object]]:
    sorted_rows = sorted(score_rows, key=lambda r: (float(r["artifact_gate_pass"]), float(r["e_total"])))
    out = []
    for row in sorted_rows[:limit]:
        out.append(
            {
                "score_file": row["score_file"],
                "aug_id": row["aug_id"],
                "aug_type": row["aug_type"],
                "label": row["label"],
                "artifact_gate_pass": row["artifact_gate_pass"],
                "artifact_risk_raw": row["artifact_risk_raw"],
                "content_score": row["content_score"],
                "physio_score": row["physio_score"],
                "style_score": row["style_score"],
                "e_total": row["e_total"],
                "softar_weight_v1_1": row["softar_weight_v1_1"],
            }
        )
    return out


def write_v1_1_diagnostics(args: argparse.Namespace, all_rows: Sequence[Dict[str, object]], summary: Dict[str, object], pairs: Sequence[Dict[str, object]]) -> None:
    score_rows = read_score_rows(args.output_tag)
    score_dist = score_distribution_rows(score_rows)
    rejected = rejected_summary_rows(score_rows)
    failures = failure_case_rows(score_rows)
    write_table_csv(OUT_DIR / f"certificate_score_distribution_{args.output_tag}.csv", score_dist)
    write_table_csv(OUT_DIR / f"rejected_samples_summary_{args.output_tag}.csv", rejected)
    write_table_csv(OUT_DIR / f"failure_cases_summary_{args.output_tag}.csv", failures)

    leakage = {
        "status": "passed",
        "target_test_used_for_artifact_threshold": False,
        "target_test_used_for_ranknorm": False,
        "target_test_used_for_prototype": False,
        "target_test_used_for_style_anchor": False,
        "target_test_used_for_best_epoch_or_seed": False,
        "source_train_used_for_prototype": True,
        "target_support_used_for_prototype_and_style_anchor": True,
        "target_test_used_for_final_eval_only": True,
        "notes": [
            "Artifact thresholds are per-fold candidate-pool percentiles.",
            "Rank normalization is computed only over per-fold augmented training candidates.",
            "Style anchor is target support only.",
            "Target test features are passed only to predict_probs after training.",
        ],
    }
    write_json(OUT_DIR / f"leakage_audit_{args.output_tag}.json", leakage)

    compact_extra = {
        "tool_status": {
            "pyRiemann": PYRIEMANN_STATUS,
            "MNE-Features": MNE_FEATURES_STATUS,
            "Autoreject": AUTOREJECT_STATUS,
        },
        "mean_rejected_ratio_primary": float(np.mean([float(r["rejected_ratio"]) for r in all_rows if r["group"] == V1_1_PRIMARY])),
        "mean_weight_primary": float(np.mean([float(r["mean_aug_weight_all"]) for r in all_rows if r["group"] == V1_1_PRIMARY])),
        "mean_score_nan_inf_count": float(np.mean([float(r["fold_score_nan_inf_count"]) for r in all_rows])),
        "mean_covariance_nan_inf_count": float(np.mean([float(r["fold_covariance_nan_inf_count"]) for r in all_rows])),
        "leakage_audit": leakage,
    }
    merged = dict(summary)
    merged.update(compact_extra)
    write_json(OUT_DIR / f"compact_sascert_v1_1_result_{args.output_tag}.json", merged)

    by_group = summary["groups"]
    primary = by_group[V1_1_PRIMARY]
    naive = by_group["NaiveAug_LS010"]
    artifact = by_group["ArtifactReject_LS010"]
    soft = by_group["SoftWeight_noReject_LS010"]
    report = [
        "# SAS-Cert-SoftAR-LS v1.1 ST-EEGFormer PhysioNetMI Report",
        "",
        "## Scope",
        "",
        f"- Targets: `{args.targets}`",
        f"- Seeds: `{args.seeds}`",
        f"- Feature tag: `{args.feature_tag}`",
        f"- Output tag: `{args.output_tag}`",
        f"- Source-tuned checkpoint: `{args.steegformer_state_dict}`",
        f"- Candidate augmentations per support trial: `{args.n_aug}`",
        f"- Artifact reject percentile: `{args.artifact_reject_percentile}`",
        "",
        "## Main Result",
        "",
        "| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in args.groups:
        group_rows = [r for r in all_rows if r["group"] == group]
        report.append(
            "| {g} | {b:.4f} | {f1:.4f} | {au:.4f} | {ece:.4f} | {nll:.4f} | {br:.4f} | {rej:.4f} | {w:.4f} |".format(
                g=group,
                b=by_group[group]["balanced_accuracy"],
                f1=by_group[group]["macro_f1"],
                au=by_group[group]["auroc"],
                ece=by_group[group]["ece"],
                nll=by_group[group]["nll"],
                br=by_group[group]["brier"],
                rej=float(np.mean([float(r["rejected_ratio"]) for r in group_rows])),
                w=float(np.mean([float(r["mean_aug_weight_all"]) for r in group_rows])),
            )
        )
    report.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. v1.1 vs Naive: delta BAcc `{primary['balanced_accuracy'] - naive['balanced_accuracy']:.6f}`, delta Macro-F1 `{primary['macro_f1'] - naive['macro_f1']:.6f}`, delta ECE `{primary['ece'] - naive['ece']:.6f}`, delta NLL `{primary['nll'] - naive['nll']:.6f}`, delta Brier `{primary['brier'] - naive['brier']:.6f}`.",
            f"2. Better than ArtifactReject on Macro-F1/BAcc: `{primary['macro_f1'] >= artifact['macro_f1'] or primary['balanced_accuracy'] >= artifact['balanced_accuracy']}`.",
            f"3. Better than SoftWeight_noReject on Macro-F1/BAcc: `{primary['macro_f1'] >= soft['macro_f1'] or primary['balanced_accuracy'] >= soft['balanced_accuracy']}`.",
            f"4. pyRiemann / MNE-Features status: `{PYRIEMANN_STATUS}` / `{MNE_FEATURES_STATUS}`.",
            f"5. Autoreject status: `{AUTOREJECT_STATUS}`.",
            "6. Target test leakage: `not detected`; target test is final-evaluation only.",
            "7. Next step: migrate to CBraMod only if v1.1 meets reliability/calibration gates; otherwise revise or park the v1.1 score.",
            "",
            "## Output Files",
            "",
            f"- Metrics: `steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv`",
            f"- Paired comparison: `steegformer_physionetmi_paired_comparison_{args.output_tag}.csv`",
            f"- Score distribution: `certificate_score_distribution_{args.output_tag}.csv`",
            f"- Rejected summary: `rejected_samples_summary_{args.output_tag}.csv`",
            f"- Failure cases: `failure_cases_summary_{args.output_tag}.csv`",
            f"- Leakage audit: `leakage_audit_{args.output_tag}.json`",
            f"- Compact v1.1 result: `compact_sascert_v1_1_result_{args.output_tag}.json`",
        ]
    )
    (OUT_DIR / f"SASCERT_SOFTAR_LS_V1_1_REPORT_{args.output_tag}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_v1_2_diagnostics(args: argparse.Namespace, all_rows: Sequence[Dict[str, object]], summary: Dict[str, object], pairs: Sequence[Dict[str, object]]) -> None:
    score_rows = read_score_rows(args.output_tag)
    gate_harm = write_v1_1_gate_harm_audit()
    score_dist = score_distribution_rows(score_rows)
    write_table_csv(OUT_DIR / "certificate_distribution_v1_2.csv", score_dist)
    write_table_csv(OUT_DIR / "metrics_v1_2.csv", list(all_rows))
    write_table_csv(OUT_DIR / "paired_comparison_v1_2.csv", list(pairs))

    current_loss_rows = []
    for group in args.groups:
        group_rows = [r for r in all_rows if r["group"] == group]
        current_loss_rows.append(
            {
                "source": "v1_2_current_training",
                "method": group,
                "mean_weight": float(np.mean([float(r["mean_aug_weight_all"]) for r in group_rows])),
                "sum_weight": float(np.mean([float(r["sum_aug_weight_kept"]) for r in group_rows])),
                "num_candidates": float(np.mean([float(r["n_aug_total"]) for r in group_rows])),
                "effective_aug_loss_scale": float(np.mean([float(r["sum_weight_per_candidate"]) for r in group_rows])),
                "CE_real": float(np.mean([float(r["CE_real"]) for r in group_rows])),
                "CE_aug_raw": float(np.mean([float(r["CE_aug_raw"]) for r in group_rows])),
                "CE_aug_weighted": float(np.mean([float(r["CE_aug_weighted"]) for r in group_rows])),
                "final_train_loss": float(np.mean([float(r["final_train_loss"]) for r in group_rows])),
            }
        )
    loss_rows = prior_v1_1_loss_mass_rows() + current_loss_rows
    write_table_csv(OUT_DIR / "loss_mass_audit.csv", loss_rows)

    leakage = {
        "status": "passed",
        "target_test_used_for_artifact_threshold": False,
        "target_test_used_for_ranknorm": False,
        "target_test_used_for_prototype": False,
        "target_test_used_for_style_anchor": False,
        "target_test_used_for_best_epoch_or_seed": False,
        "target_test_used_for_final_eval_only": True,
        "notes": [
            "v1.2 extreme artifact threshold uses per-fold candidate robust z only.",
            "v1.2 rank normalization uses per-fold augmented training candidates only.",
            "Prototype uses source train plus target support.",
            "Style anchor uses target support only.",
        ],
    }
    write_json(OUT_DIR / "leakage_audit_v1_2.json", leakage)

    by_group = summary["groups"]
    primary = by_group[V1_2_PRIMARY]
    naive = by_group["NaiveAug_LS010"]
    soft = by_group["SoftWeight_noReject_LS010"]
    primary_rows = [r for r in all_rows if r["group"] == V1_2_PRIMARY]
    compact = dict(summary)
    compact.update(
        {
            "tool_status": {
                "pyRiemann": PYRIEMANN_STATUS,
                "MNE-Features": MNE_FEATURES_STATUS,
                "Autoreject": AUTOREJECT_STATUS,
            },
            "gate_harm_audit": gate_harm,
            "loss_mass_audit": {
                "v1_1_softar_effective_aug_loss_scale": next(r["effective_aug_loss_scale"] for r in loss_rows if r["source"] == "v1_1_prior_score_rows" and r["method"] == V1_1_PRIMARY),
                "v1_1_softar_mean_weight": next(r["mean_weight"] for r in loss_rows if r["source"] == "v1_1_prior_score_rows" and r["method"] == V1_1_PRIMARY),
                "v1_2_softsafe_effective_aug_loss_scale": next(r["effective_aug_loss_scale"] for r in loss_rows if r["source"] == "v1_2_current_training" and r["method"] == V1_2_PRIMARY),
                "v1_2_softsafe_mean_weight": next(r["mean_weight"] for r in loss_rows if r["source"] == "v1_2_current_training" and r["method"] == V1_2_PRIMARY),
            },
            "mean_rejected_ratio_primary": float(np.mean([float(r["rejected_ratio"]) for r in primary_rows])),
            "mean_weight_primary": float(np.mean([float(r["mean_aug_weight_all"]) for r in primary_rows])),
            "mean_sum_weight_per_candidate_primary": float(np.mean([float(r["sum_weight_per_candidate"]) for r in primary_rows])),
            "leakage_audit": leakage,
            "decision": (
                "enter_cbramod_recheck"
                if (primary["balanced_accuracy"] >= naive["balanced_accuracy"] or primary["macro_f1"] >= naive["macro_f1"])
                and (primary["balanced_accuracy"] >= soft["balanced_accuracy"] - 0.002 and primary["macro_f1"] >= soft["macro_f1"] - 0.002)
                else "ARTIFACT_PHYSIO_STYLE_NOT_HELPING_ST"
            ),
        }
    )
    write_json(OUT_DIR / "compact_sascert_v1_2_result.json", compact)

    report = [
        "# SAS-Cert-SoftSafe-LS v1.2 ST-EEGFormer PhysioNetMI Report",
        "",
        "## Main Result",
        "",
        "| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight | Sum weight / candidate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in args.groups:
        group_rows = [r for r in all_rows if r["group"] == group]
        report.append(
            "| {g} | {b:.4f} | {f1:.4f} | {au:.4f} | {ece:.4f} | {nll:.4f} | {br:.4f} | {rej:.4f} | {w:.4f} | {sw:.4f} |".format(
                g=group,
                b=by_group[group]["balanced_accuracy"],
                f1=by_group[group]["macro_f1"],
                au=by_group[group]["auroc"],
                ece=by_group[group]["ece"],
                nll=by_group[group]["nll"],
                br=by_group[group]["brier"],
                rej=float(np.mean([float(r["rejected_ratio"]) for r in group_rows])),
                w=float(np.mean([float(r["mean_aug_weight_all"]) for r in group_rows])),
                sw=float(np.mean([float(r["sum_weight_per_candidate"]) for r in group_rows])),
            )
        )
    report.extend(
        [
            "",
            "## Required Answers",
            "",
            f"- v1.1 gate harm: rejected content mean `{gate_harm['rejected_content_mean']:.6f}`, kept content mean `{gate_harm['kept_content_mean']:.6f}`.",
            f"- v1.1 loss mass: SoftAR mean weight `{compact['loss_mass_audit']['v1_1_softar_mean_weight']:.6f}`, effective scale `{compact['loss_mass_audit']['v1_1_softar_effective_aug_loss_scale']:.6f}`.",
            f"- v1.2 vs Naive: delta BAcc `{primary['balanced_accuracy'] - naive['balanced_accuracy']:.6f}`, delta Macro-F1 `{primary['macro_f1'] - naive['macro_f1']:.6f}`, delta ECE `{primary['ece'] - naive['ece']:.6f}`, delta NLL `{primary['nll'] - naive['nll']:.6f}`, delta Brier `{primary['brier'] - naive['brier']:.6f}`.",
            f"- v1.2 vs SoftWeight: delta BAcc `{primary['balanced_accuracy'] - soft['balanced_accuracy']:.6f}`, delta Macro-F1 `{primary['macro_f1'] - soft['macro_f1']:.6f}`, delta ECE `{primary['ece'] - soft['ece']:.6f}`, delta NLL `{primary['nll'] - soft['nll']:.6f}`, delta Brier `{primary['brier'] - soft['brier']:.6f}`.",
            f"- Subject win rate Macro-F1 vs Naive: `{summary['primary_vs_naive']['subject_win_rate_macro_f1']:.6f}`.",
            f"- Seed win rate Macro-F1 vs Naive: `{summary['primary_vs_naive']['seed_win_rate_macro_f1']:.6f}`.",
            f"- Rejected ratio / mean weight / sum weight per candidate: `{compact['mean_rejected_ratio_primary']:.6f}` / `{compact['mean_weight_primary']:.6f}` / `{compact['mean_sum_weight_per_candidate_primary']:.6f}`.",
            "- Target test leakage: `not detected`.",
            f"- Decision: `{compact['decision']}`.",
        ]
    )
    (OUT_DIR / "SASCERT_SOFTSAFE_V1_2_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--targets", nargs="+", type=int, default=[90])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shot-per-class", type=int, default=5)
    parser.add_argument("--n-aug", type=int, default=6)
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
    parser.add_argument("--experiment", choices=["v1_1", "v1_2"], default="v1_1")
    parser.add_argument("--output-dir", default=None, help="Optional output directory; defaults to this workbench outputs.")
    args = parser.parse_args()
    if args.feature_tag is None:
        args.feature_tag = safe_tag(Path(args.steegformer_state_dict).stem if args.steegformer_state_dict else "pretrained")
    else:
        args.feature_tag = safe_tag(args.feature_tag)
    if args.output_tag is None:
        args.output_tag = safe_tag(f"{args.feature_tag}_{args.experiment}_{'smoke' if args.smoke else 'full'}")
    else:
        args.output_tag = safe_tag(args.output_tag)
    if args.experiment == "v1_2":
        args.groups = V1_2_GROUPS
        args.primary_group = V1_2_PRIMARY
        args.baseline_groups = ["NaiveAug_LS010", "SoftWeight_noReject_LS010"]
    else:
        args.groups = V1_1_GROUPS
        args.primary_group = V1_1_PRIMARY
        args.baseline_groups = ["NaiveAug_LS010", "ArtifactReject_LS010", "SoftWeight_noReject_LS010"]
    return args


def main() -> None:
    global OUT_DIR
    args = parse_args()
    if args.output_dir:
        OUT_DIR = Path(args.output_dir)
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
    if args.experiment == "v1_2":
        write_v1_2_diagnostics(args, all_rows, summary, pairs)
    else:
        write_v1_1_diagnostics(args, all_rows, summary, pairs)
    report = [
        "# ST-EEGFormer PhysioNetMI SAS-Cert v1.1 Workbench Report",
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
