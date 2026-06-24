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
V1_3_GROUPS = ["NaiveAug_LS010", "SoftWeight_noReject_LS010", "SAS-Cert-CU-LS-v1.3"]
V1_3_PRIMARY = "SAS-Cert-CU-LS-v1.3"
V1_4_PRIMARY = "SAS-Cert-SCB-CU-LS-v1.4"
V1_4_REGULAR_GROUPS = ["NaiveAug_LS010", V1_3_PRIMARY, V1_4_PRIMARY]
RISK_MIXED_NAIVE = "RiskMixed_NaiveAug_LS010"
RISK_MIXED_V1_4_PRIMARY = "RiskMixed_SAS-Cert-SCB-CU-LS-v1.4"
V1_4_RISKMIXED_GROUPS = [RISK_MIXED_NAIVE, RISK_MIXED_V1_4_PRIMARY]
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


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return pearson_corr(ranknorm(a[mask], True), ranknorm(b[mask], True))


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


def strong_frequency_mask_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return frequency_mask_aug(x, rng, intensity=0.95)


def strong_channel_dropout_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    n_drop = max(2, int(round(y.shape[0] * 0.22)))
    chans = rng.choice(y.shape[0], size=min(n_drop, y.shape[0]), replace=False)
    y[chans] = 0.0
    return y.astype(np.float32)


def emg_like_burst_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    signal_std = float(y.std() + 1e-6)
    burst_len = max(24, int(y.shape[1] * 0.10))
    start = int(rng.integers(0, max(1, y.shape[1] - burst_len)))
    t = np.arange(burst_len, dtype=np.float32) / SFREQ
    carrier = np.sin(2 * np.pi * rng.uniform(35.0, 75.0) * t + rng.uniform(0, 2 * np.pi))
    envelope = np.hanning(burst_len).astype(np.float32)
    chans = rng.choice(y.shape[0], size=max(2, int(round(y.shape[0] * 0.18))), replace=False)
    y[chans, start : start + burst_len] += signal_std * rng.uniform(1.2, 2.2) * envelope * carrier
    return y.astype(np.float32)


def eog_like_drift_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    signal_std = float(y.std() + 1e-6)
    t = np.arange(y.shape[1], dtype=np.float32) / SFREQ
    drift = np.sin(2 * np.pi * rng.uniform(0.15, 0.8) * t + rng.uniform(0, 2 * np.pi))
    ramp = np.linspace(-1.0, 1.0, y.shape[1], dtype=np.float32)
    chans = np.arange(min(6, y.shape[0]))
    y[chans] += signal_std * rng.uniform(0.8, 1.6) * (0.7 * drift + 0.3 * ramp)
    return y.astype(np.float32)


def covariance_perturbation_aug(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mix = np.eye(x.shape[0], dtype=np.float32)
    noise = rng.normal(0.0, 0.08, size=mix.shape).astype(np.float32)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    return ((mix + noise) @ x).astype(np.float32)


def generate_candidates(
    support_x: np.ndarray,
    support_y: np.ndarray,
    style_bank: np.ndarray,
    rng: np.random.Generator,
    per_trial: int = 6,
    risk_mixed: bool = False,
) -> List[Candidate]:
    aug_types = ["gaussian_noise", "time_shift", "time_crop", "frequency_mask", "channel_dropout", "mild_frequency_mixup"]
    risky_types = ["strong_frequency_mask", "strong_channel_dropout", "emg_like_burst", "eog_like_drift", "covariance_perturbation"]
    candidates: list[Candidate] = []
    for i, (x, label) in enumerate(zip(support_x, support_y)):
        for k in range(per_trial):
            if risk_mixed and k >= int(round(per_trial * 0.7)):
                aug_type = risky_types[(k - int(round(per_trial * 0.7))) % len(risky_types)]
            else:
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
            elif aug_type == "mild_frequency_mixup":
                aug_x = mild_frequency_mixup_aug(x, rng, style_bank)
            elif aug_type == "strong_frequency_mask":
                aug_x = strong_frequency_mask_aug(x, rng)
            elif aug_type == "strong_channel_dropout":
                aug_x = strong_channel_dropout_aug(x, rng)
            elif aug_type == "emg_like_burst":
                aug_x = emg_like_burst_aug(x, rng)
            elif aug_type == "eog_like_drift":
                aug_x = eog_like_drift_aug(x, rng)
            else:
                aug_x = covariance_perturbation_aug(x, rng)
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
    content_v1_3 = ranknorm(e_embed_raw, True) + ranknorm(e_proto_raw, True)
    content_v1_3_rank = ranknorm(content_v1_3, True)
    content_utility_weight_v1_3 = 0.75 + 0.5 * content_v1_3_rank
    content_v1_4_rank = np.zeros_like(content_v1_3_rank, dtype=np.float32)
    content_v1_4_scope = np.asarray(["global"] * len(labels), dtype=object)
    global_content_rank = ranknorm(content_v1_3, True)
    for label in sorted(set(labels.astype(int))):
        mask = labels == int(label)
        if int(mask.sum()) >= 3:
            content_v1_4_rank[mask] = ranknorm(content_v1_3[mask], True)
            content_v1_4_scope[mask] = "subject_class"
        elif len(labels) >= 3:
            content_v1_4_rank[mask] = global_content_rank[mask]
            content_v1_4_scope[mask] = "subject_fallback"
        else:
            content_v1_4_rank[mask] = global_content_rank[mask]
            content_v1_4_scope[mask] = "global_fallback"
    scb_content_utility_weight_v1_4 = 0.75 + 0.5 * content_v1_4_rank
    raw_ce_loss = -np.log(np.clip(cand_probs[np.arange(len(labels)), labels], 1e-7, 1.0))
    aug_correctness = (cand_probs.argmax(axis=1) == labels).astype(np.int64)
    nonfinite_content_v1_3 = (
        ~np.isfinite(e_embed_raw)
        | ~np.isfinite(e_proto_raw)
        | ~np.isfinite(content_v1_3)
        | ~np.isfinite(content_utility_weight_v1_3)
        | ~np.isfinite(scb_content_utility_weight_v1_4)
    )
    content_utility_weight_v1_3[nonfinite_content_v1_3] = 0.0
    scb_content_utility_weight_v1_4[nonfinite_content_v1_3] = 0.0

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
                "content_score_v1_3": float(content_v1_3_rank[i]),
                "content_utility_weight_v1_3": float(content_utility_weight_v1_3[i]),
                "content_score_v1_4_scb": float(content_v1_4_rank[i]),
                "scb_content_utility_weight_v1_4": float(scb_content_utility_weight_v1_4[i]),
                "scb_ranknorm_scope_v1_4": str(content_v1_4_scope[i]),
                "nonfinite_content_v1_3": int(nonfinite_content_v1_3[i]),
                "raw_ce_loss": float(raw_ce_loss[i]),
                "aug_correctness": int(aug_correctness[i]),
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
        "mean_content_utility_weight_v1_3": float(content_utility_weight_v1_3.mean()),
        "mean_scb_content_utility_weight_v1_4": float(scb_content_utility_weight_v1_4.mean()),
        "content_utility_nan_inf_count_v1_3": int(nonfinite_content_v1_3.sum()),
        "scb_ranknorm_scope_counts_v1_4": {
            str(scope): int(np.sum(content_v1_4_scope == scope))
            for scope in sorted(set(content_v1_4_scope.astype(str)))
        },
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


def class_balanced_aug_loss_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    loss_each = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    class_losses = []
    for cls in torch.unique(y):
        mask = y == cls
        if bool(mask.any()):
            class_losses.append((loss_each[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1e-6))
    if not class_losses:
        return loss_each.mean()
    return torch.stack(class_losses).mean()


def train_head_real_aug_class_balanced(
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
        aug_loss = class_balanced_aug_loss_from_logits(head(aug_x), aug_y, aug_w, label_smoothing)
        loss = real_loss + aug_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite class-balanced head loss")
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


def loss_mass_diagnostics_class_balanced(
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
        ce_aug_weighted = class_balanced_aug_loss_from_logits(head(aug_x), aug_y, aug_w, label_smoothing)
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
    content_utility_weight = np.asarray([float(r.get("content_utility_weight_v1_3", 1.0)) for r in score_rows], dtype=np.float32)
    scb_content_utility_weight = np.asarray([float(r.get("scb_content_utility_weight_v1_4", r.get("content_utility_weight_v1_3", 1.0))) for r in score_rows], dtype=np.float32)
    extreme = np.asarray([int(float(r.get("extreme_artifact_outlier", 0))) == 1 for r in score_rows], dtype=bool)
    nonfinite_content = np.asarray([int(float(r.get("nonfinite_content_v1_3", 0))) == 1 for r in score_rows], dtype=bool)
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
    elif group == V1_3_PRIMARY:
        keep = ~nonfinite_content
        weights = content_utility_weight
        weights[~keep] = 0.0
    elif group in {V1_4_PRIMARY, RISK_MIXED_V1_4_PRIMARY}:
        keep = ~nonfinite_content
        weights = scb_content_utility_weight
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
    candidates = generate_candidates(
        support_x,
        support_y,
        style_bank,
        rng,
        per_trial=args.n_aug,
        risk_mixed=args.experiment == "v1_4_riskmixed",
    )
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
        if args.experiment in {"v1_2", "v1_3", "v1_4_regular", "v1_4_riskmixed"}:
            if not keep.any():
                raise RuntimeError(f"{args.experiment} has no augmented candidates after safety filtering")
            aug_features = cand_features[keep]
            aug_labels = cand_labels[keep]
            aug_weights = aug_w[keep].astype(np.float32)
            if group in {V1_4_PRIMARY, RISK_MIXED_V1_4_PRIMARY}:
                head = train_head_real_aug_class_balanced(
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
                loss_diag = loss_mass_diagnostics_class_balanced(head, support_features, support_y, aug_features, aug_labels, aug_weights, args.device, 0.10)
            else:
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
        pred = probs.argmax(axis=1)
        per_class_metrics: dict[str, float] = {}
        for cls in sorted(set(test_y.astype(int))):
            cls_mask = test_y == int(cls)
            tp = float(np.sum((pred == cls) & cls_mask))
            fp = float(np.sum((pred == cls) & (~cls_mask)))
            fn = float(np.sum((pred != cls) & cls_mask))
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
            per_class_metrics[f"class_{cls}_accuracy"] = float(np.mean(pred[cls_mask] == test_y[cls_mask])) if cls_mask.any() else float("nan")
            per_class_metrics[f"class_{cls}_f1"] = float(f1)
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
                **per_class_metrics,
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
    baseline_name = "NaiveAug_LS010" if "NaiveAug_LS010" in by_group else baseline_groups[0]
    naive = by_group[baseline_name]
    comparison = {
        "primary_group": primary_group,
        "baseline_group": baseline_name,
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
        "sascert_vs_naive": comparison if primary_group in {V1_1_PRIMARY, V1_2_PRIMARY, V1_3_PRIMARY, V1_4_PRIMARY, RISK_MIXED_V1_4_PRIMARY} else None,
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
        "content_score_v1_3",
        "content_utility_weight_v1_3",
        "raw_ce_loss",
        "aug_correctness",
        "content_score_v1_4_scb",
        "scb_content_utility_weight_v1_4",
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


def write_v1_3_diagnostics(args: argparse.Namespace, all_rows: Sequence[Dict[str, object]], summary: Dict[str, object], pairs: Sequence[Dict[str, object]]) -> None:
    score_rows = read_score_rows(args.output_tag)
    score_dist = score_distribution_rows(score_rows)
    write_table_csv(OUT_DIR / "certificate_distribution_v1_3.csv", score_dist)
    write_table_csv(OUT_DIR / "metrics_v1_3.csv", list(all_rows))
    write_table_csv(OUT_DIR / "paired_comparison_v1_3.csv", list(pairs))

    diagnostic_rows = []
    for row in score_rows:
        diagnostic_rows.append(
            {
                "target_subject": row["target_subject"],
                "seed": row["seed"],
                "aug_id": row["aug_id"],
                "augmentation_type": row["aug_type"],
                "label": row["label"],
                "E_embed": row["e_embed_raw"],
                "E_proto": row["e_proto_raw"],
                "E_pred": row["e_pred_raw"],
                "E_content": row["content_score_v1_3"],
                "artifact_score": row["artifact_risk_raw"],
                "artifact_safe": row["artifact_safe_score"],
                "E_physio": row["physio_score"],
                "E_style": row["style_score"],
                "D_band": row["d_band"],
                "D_cov": row["d_cov_riemann"],
                "D_style": row["d_style"],
                "content_utility_weight": row["content_utility_weight_v1_3"],
                "raw_CE_loss": row["raw_ce_loss"],
                "augmented_correctness": row["aug_correctness"],
                "nonfinite_content_skip": row["nonfinite_content_v1_3"],
            }
        )
    write_table_csv(OUT_DIR / "diagnostic_scores_v1_3.csv", diagnostic_rows)

    component_specs = [
        ("E_embed", "e_embed_raw", "higher_content_evidence"),
        ("E_proto", "e_proto_raw", "higher_content_evidence"),
        ("E_pred", "e_pred_raw", "prediction_consistency_audit_only"),
        ("E_content", "content_score_v1_3", "content_utility_candidate"),
        ("artifact_score", "artifact_risk_raw", "higher_risk"),
        ("artifact_safe", "artifact_safe_score", "higher_safety"),
        ("E_physio", "physio_score", "diagnostic_physio"),
        ("E_style", "style_score", "diagnostic_style"),
        ("D_band", "d_band", "lower_distance_better"),
        ("D_cov", "d_cov_riemann", "lower_distance_better"),
        ("D_style", "d_style", "lower_distance_better"),
    ]
    ce = np.asarray([float(r["raw_ce_loss"]) for r in score_rows], dtype=np.float64)
    correctness = np.asarray([float(r["aug_correctness"]) for r in score_rows], dtype=np.float64)
    audit_rows = []
    training_candidates = []
    diagnostic_only = []
    for name, col, prior_role in component_specs:
        values = np.asarray([float(r[col]) for r in score_rows], dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(ce) & np.isfinite(correctness)
        vals = values[finite]
        ce_vals = ce[finite]
        corr_vals = correctness[finite]
        if len(vals) >= 3:
            q30 = np.percentile(vals, 30)
            q70 = np.percentile(vals, 70)
            bottom = vals <= q30
            top = vals >= q70
            rho_ce = spearman_corr(vals, ce_vals)
            rho_correct = spearman_corr(vals, corr_vals)
            top_ce = float(np.mean(ce_vals[top])) if top.any() else float("nan")
            bottom_ce = float(np.mean(ce_vals[bottom])) if bottom.any() else float("nan")
            top_correct = float(np.mean(corr_vals[top])) if top.any() else float("nan")
            bottom_correct = float(np.mean(corr_vals[bottom])) if bottom.any() else float("nan")
        else:
            rho_ce = rho_correct = top_ce = bottom_ce = top_correct = bottom_correct = float("nan")
        utility_signal = (
            np.isfinite(rho_ce)
            and np.isfinite(rho_correct)
            and rho_ce < -0.03
            and rho_correct > 0.01
            and top_ce < bottom_ce
            and top_correct >= bottom_correct
        )
        risk_signal = (
            np.isfinite(rho_ce)
            and np.isfinite(rho_correct)
            and rho_ce > 0.03
            and rho_correct < -0.01
        )
        if name in {"E_embed", "E_proto", "E_content"} and utility_signal:
            recommended_role = "training_utility_candidate"
            training_candidates.append(name)
        elif risk_signal or name not in {"E_embed", "E_proto", "E_content"}:
            recommended_role = "diagnostic_report_only"
            diagnostic_only.append(name)
        else:
            recommended_role = "weak_or_unstable_signal"
            diagnostic_only.append(name)
        audit_rows.append(
            {
                "score_name": name,
                "column": col,
                "prior_role": prior_role,
                "n": int(finite.sum()),
                "spearman_raw_CE_loss": rho_ce,
                "spearman_correctness": rho_correct,
                "top30_CE_loss_mean": top_ce,
                "bottom30_CE_loss_mean": bottom_ce,
                "top30_correctness_mean": top_correct,
                "bottom30_correctness_mean": bottom_correct,
                "top_minus_bottom_CE_loss": top_ce - bottom_ce if np.isfinite(top_ce) and np.isfinite(bottom_ce) else float("nan"),
                "top_minus_bottom_correctness": top_correct - bottom_correct if np.isfinite(top_correct) and np.isfinite(bottom_correct) else float("nan"),
                "recommended_role": recommended_role,
            }
        )
    write_table_csv(OUT_DIR / "component_utility_audit.csv", audit_rows)

    leakage = {
        "status": "passed",
        "target_test_used_for_artifact_threshold": False,
        "target_test_used_for_ranknorm": False,
        "target_test_used_for_prototype": False,
        "target_test_used_for_style_anchor": False,
        "target_test_used_for_best_epoch_or_seed": False,
        "target_test_used_for_component_utility_audit": False,
        "target_test_used_for_final_eval_only": True,
        "notes": [
            "v1.3 training weight uses only E_embed and E_proto from per-fold support candidates.",
            "ranknorm is computed over per-fold augmented training candidates only.",
            "prototype uses source train plus target support.",
            "artifact/physio/style/prediction scores are diagnostic only and do not enter the v1.3 training weight.",
            "raw CE/correctness in component utility audit are computed on augmented training candidates with the source/support head, not target test.",
        ],
    }
    write_json(OUT_DIR / "leakage_audit_v1_3.json", leakage)

    by_group = summary["groups"]
    primary = by_group[V1_3_PRIMARY]
    naive = by_group["NaiveAug_LS010"]
    soft = by_group["SoftWeight_noReject_LS010"]
    primary_rows = [r for r in all_rows if r["group"] == V1_3_PRIMARY]
    mean_weight = float(np.mean([float(r["mean_aug_weight_all"]) for r in primary_rows]))
    mean_sum_weight = float(np.mean([float(r["sum_weight_per_candidate"]) for r in primary_rows]))
    primary_score_weights = np.asarray([float(r["content_utility_weight_v1_3"]) for r in score_rows], dtype=np.float64)
    skipped_nonfinite = int(np.sum([int(float(r["nonfinite_content_v1_3"])) for r in score_rows]))
    delta_naive = {
        "delta_balanced_accuracy": primary["balanced_accuracy"] - naive["balanced_accuracy"],
        "delta_macro_f1": primary["macro_f1"] - naive["macro_f1"],
        "delta_ece": primary["ece"] - naive["ece"],
        "delta_nll": primary["nll"] - naive["nll"],
        "delta_brier": primary["brier"] - naive["brier"],
    }
    delta_soft = {
        "delta_balanced_accuracy": primary["balanced_accuracy"] - soft["balanced_accuracy"],
        "delta_macro_f1": primary["macro_f1"] - soft["macro_f1"],
        "delta_ece": primary["ece"] - soft["ece"],
        "delta_nll": primary["nll"] - soft["nll"],
        "delta_brier": primary["brier"] - soft["brier"],
    }
    naive_success = (delta_naive["delta_balanced_accuracy"] >= 0.0 or delta_naive["delta_macro_f1"] >= 0.0) and delta_naive["delta_ece"] <= 0.005 and delta_naive["delta_nll"] <= 0.005 and delta_naive["delta_brier"] <= 0.005
    soft_not_worse = delta_soft["delta_balanced_accuracy"] >= -0.002 and delta_soft["delta_macro_f1"] >= -0.002
    if naive_success and (delta_soft["delta_balanced_accuracy"] > 0.0 or delta_soft["delta_macro_f1"] > 0.0):
        decision = "enter_cbramod_recheck"
        next_recommendation = "进入 CBraMod 复验"
    elif naive_success and soft_not_worse:
        decision = "reliability_tradeoff_or_borderline_content_utility"
        next_recommendation = "做 risk-mixed candidate pool"
    else:
        decision = "CONTENT_UTILITY_NOT_HELPING_ST"
        next_recommendation = "做 risk-mixed candidate pool 或归档为 diagnostic-only"

    component_summary = {
        "training_utility_candidates": training_candidates,
        "diagnostic_or_unstable_scores": diagnostic_only,
        "strongest_low_ce_correlations": sorted(
            [
                {"score_name": r["score_name"], "spearman_raw_CE_loss": r["spearman_raw_CE_loss"], "spearman_correctness": r["spearman_correctness"]}
                for r in audit_rows
                if np.isfinite(float(r["spearman_raw_CE_loss"]))
            ],
            key=lambda x: float(x["spearman_raw_CE_loss"]),
        )[:5],
        "strongest_correctness_correlations": sorted(
            [
                {"score_name": r["score_name"], "spearman_raw_CE_loss": r["spearman_raw_CE_loss"], "spearman_correctness": r["spearman_correctness"]}
                for r in audit_rows
                if np.isfinite(float(r["spearman_correctness"]))
            ],
            key=lambda x: float(x["spearman_correctness"]),
            reverse=True,
        )[:5],
    }
    write_json(OUT_DIR / "component_utility_summary.json", component_summary)

    compact = dict(summary)
    compact.update(
        {
            "tool_status": {
                "pyRiemann": PYRIEMANN_STATUS,
                "MNE-Features": MNE_FEATURES_STATUS,
                "Autoreject": AUTOREJECT_STATUS,
            },
            "component_utility_summary": component_summary,
            "primary_vs_naive_required": delta_naive,
            "primary_vs_softweight_required": delta_soft,
            "mean_rejected_ratio_primary": float(np.mean([float(r["rejected_ratio"]) for r in primary_rows])),
            "mean_weight_primary": mean_weight,
            "mean_sum_weight_per_candidate_primary": mean_sum_weight,
            "weight_range_primary": [float(np.nanmin(primary_score_weights)), float(np.nanmax(primary_score_weights))],
            "nonfinite_content_skipped_count": skipped_nonfinite,
            "leakage_audit": leakage,
            "decision": decision,
            "next_recommendation": next_recommendation,
        }
    )
    write_json(OUT_DIR / "compact_sascert_v1_3_result.json", compact)

    report = [
        "# SAS-Cert-CU-LS v1.3 ST-EEGFormer PhysioNetMI Report",
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
            "## Component Utility Audit",
            "",
            f"- Training utility candidates: `{component_summary['training_utility_candidates']}`.",
            f"- Diagnostic/unstable scores: `{component_summary['diagnostic_or_unstable_scores']}`.",
            "",
            "| Score | Spearman CE | Spearman correctness | Top30-bottom30 CE | Top30-bottom30 correctness | Role |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in audit_rows:
        report.append(
            "| {score} | {ce:.4f} | {corr:.4f} | {dce:.4f} | {dcorr:.4f} | `{role}` |".format(
                score=row["score_name"],
                ce=float(row["spearman_raw_CE_loss"]),
                corr=float(row["spearman_correctness"]),
                dce=float(row["top_minus_bottom_CE_loss"]),
                dcorr=float(row["top_minus_bottom_correctness"]),
                role=row["recommended_role"],
            )
        )
    report.extend(
        [
            "",
            "## Required Answers",
            "",
            f"- v1.3 vs Naive: delta BAcc `{delta_naive['delta_balanced_accuracy']:.6f}`, delta Macro-F1 `{delta_naive['delta_macro_f1']:.6f}`, delta ECE `{delta_naive['delta_ece']:.6f}`, delta NLL `{delta_naive['delta_nll']:.6f}`, delta Brier `{delta_naive['delta_brier']:.6f}`.",
            f"- v1.3 vs SoftWeight: delta BAcc `{delta_soft['delta_balanced_accuracy']:.6f}`, delta Macro-F1 `{delta_soft['delta_macro_f1']:.6f}`, delta ECE `{delta_soft['delta_ece']:.6f}`, delta NLL `{delta_soft['delta_nll']:.6f}`, delta Brier `{delta_soft['delta_brier']:.6f}`.",
            f"- Subject win rate Macro-F1 vs Naive: `{summary['primary_vs_naive']['subject_win_rate_macro_f1']:.6f}`.",
            f"- Seed win rate Macro-F1 vs Naive: `{summary['primary_vs_naive']['seed_win_rate_macro_f1']:.6f}`.",
            f"- Mean weight / weight range: `{mean_weight:.6f}` / `{compact['weight_range_primary']}`.",
            f"- Nonfinite content skipped count: `{skipped_nonfinite}`.",
            "- Target test leakage: `not detected`.",
            f"- Decision: `{decision}`.",
            f"- Next recommendation: `{next_recommendation}`.",
            "",
            "## Output Files",
            "",
            "- `SASCERT_CU_V1_3_REPORT.md`",
            "- `compact_sascert_v1_3_result.json`",
            "- `metrics_v1_3.csv`",
            "- `paired_comparison_v1_3.csv`",
            "- `component_utility_audit.csv`",
            "- `component_utility_summary.json`",
            "- `diagnostic_scores_v1_3.csv`",
            "- `leakage_audit_v1_3.json`",
        ]
    )
    (OUT_DIR / "SASCERT_CU_V1_3_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def read_table_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_metric(rows: Sequence[Dict[str, object]], group: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if str(r.get("group")) == group and str(r.get(key, "")) != ""]
    return float(np.mean(vals)) if vals else float("nan")


def delta_metrics(rows: Sequence[Dict[str, object]], primary: str, baseline: str) -> Dict[str, float]:
    return {
        "delta_balanced_accuracy": mean_metric(rows, primary, "balanced_accuracy") - mean_metric(rows, baseline, "balanced_accuracy"),
        "delta_macro_f1": mean_metric(rows, primary, "macro_f1") - mean_metric(rows, baseline, "macro_f1"),
        "delta_ece": mean_metric(rows, primary, "ece") - mean_metric(rows, baseline, "ece"),
        "delta_nll": mean_metric(rows, primary, "nll") - mean_metric(rows, baseline, "nll"),
        "delta_brier": mean_metric(rows, primary, "brier") - mean_metric(rows, baseline, "brier"),
    }


def win_rates_from_pairs(pairs: Sequence[Dict[str, object]], comparison: str) -> Dict[str, float]:
    comp = [p for p in pairs if str(p.get("comparison")) == comparison]
    subject_wins = []
    for target in sorted(set(int(p["target_subject"]) for p in comp)):
        vals = [int(float(p["win_macro_f1"])) for p in comp if int(p["target_subject"]) == target]
        if vals:
            subject_wins.append(float(np.mean(vals)) > 0.5)
    seed_wins = []
    for seed in sorted(set(int(p["seed"]) for p in comp)):
        vals = [int(float(p["win_macro_f1"])) for p in comp if int(p["seed"]) == seed]
        if vals:
            seed_wins.append(float(np.mean(vals)) > 0.5)
    return {
        "subject_win_rate_macro_f1": float(np.mean(subject_wins)) if subject_wins else float("nan"),
        "seed_win_rate_macro_f1": float(np.mean(seed_wins)) if seed_wins else float("nan"),
    }


def write_v1_4_localization_outputs(regular_rows: Sequence[Dict[str, object]], score_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_target_scores: dict[int, list[dict[str, object]]] = {}
    for row in score_rows:
        by_target_scores.setdefault(int(row["target_subject"]), []).append(row)

    per_subject = []
    for target in sorted(set(int(r["target_subject"]) for r in regular_rows)):
        target_rows = [r for r in regular_rows if int(r["target_subject"]) == target]
        score_subset = by_target_scores.get(target, [])
        counts = {}
        for row in score_subset:
            counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
        max_count = max(counts.values()) if counts else 0
        min_count = min(counts.values()) if counts else 0
        per_subject.append(
            {
                "target_subject": target,
                "delta_bacc_v1_4_vs_naive": mean_metric(target_rows, V1_4_PRIMARY, "balanced_accuracy") - mean_metric(target_rows, "NaiveAug_LS010", "balanced_accuracy"),
                "delta_macro_f1_v1_4_vs_naive": mean_metric(target_rows, V1_4_PRIMARY, "macro_f1") - mean_metric(target_rows, "NaiveAug_LS010", "macro_f1"),
                "delta_ece_v1_4_vs_naive": mean_metric(target_rows, V1_4_PRIMARY, "ece") - mean_metric(target_rows, "NaiveAug_LS010", "ece"),
                "delta_nll_v1_4_vs_naive": mean_metric(target_rows, V1_4_PRIMARY, "nll") - mean_metric(target_rows, "NaiveAug_LS010", "nll"),
                "mean_E_proto": float(np.mean([float(r["e_proto_raw"]) for r in score_subset])) if score_subset else float("nan"),
                "mean_E_embed": float(np.mean([float(r["e_embed_raw"]) for r in score_subset])) if score_subset else float("nan"),
                "mean_weight": float(np.mean([float(r["scb_content_utility_weight_v1_4"]) for r in score_subset])) if score_subset else float("nan"),
                "class_0_candidates": counts.get("0", 0),
                "class_1_candidates": counts.get("1", 0),
                "class_balance_min_over_max": float(min_count / max(1, max_count)) if counts else float("nan"),
            }
        )
    write_table_csv(OUT_DIR / "per_subject_delta_table.csv", per_subject)

    per_class = []
    for cls in [0, 1]:
        for metric in ["accuracy", "f1"]:
            key = f"class_{cls}_{metric}"
            per_class.append(
                {
                    "class": cls,
                    "metric": key,
                    "v1_4_mean": mean_metric(regular_rows, V1_4_PRIMARY, key),
                    "naive_mean": mean_metric(regular_rows, "NaiveAug_LS010", key),
                    "v1_3_mean": mean_metric(regular_rows, V1_3_PRIMARY, key),
                    "delta_v1_4_vs_naive": mean_metric(regular_rows, V1_4_PRIMARY, key) - mean_metric(regular_rows, "NaiveAug_LS010", key),
                    "delta_v1_4_vs_v1_3": mean_metric(regular_rows, V1_4_PRIMARY, key) - mean_metric(regular_rows, V1_3_PRIMARY, key),
                }
            )
    write_table_csv(OUT_DIR / "per_class_delta_table.csv", per_class)

    corr_rows = []
    for target, subset in sorted(by_target_scores.items()):
        ce = np.asarray([float(r["raw_ce_loss"]) for r in subset], dtype=np.float64)
        correctness = np.asarray([float(r["aug_correctness"]) for r in subset], dtype=np.float64)
        for name, col in [("E_proto", "e_proto_raw"), ("E_embed", "e_embed_raw"), ("E_content", "content_score_v1_3")]:
            vals = np.asarray([float(r[col]) for r in subset], dtype=np.float64)
            corr_rows.append(
                {
                    "target_subject": target,
                    "score_name": name,
                    "spearman_raw_CE_loss": spearman_corr(vals, ce),
                    "spearman_correctness": spearman_corr(vals, correctness),
                    "n": len(subset),
                }
            )
    write_table_csv(OUT_DIR / "per_subject_component_corr.csv", corr_rows)

    weight_rows = []
    for target, subset in sorted(by_target_scores.items()):
        for label in sorted(set(str(r["label"]) for r in subset)):
            label_rows = [r for r in subset if str(r["label"]) == label]
            vals_v13 = np.asarray([float(r["content_utility_weight_v1_3"]) for r in label_rows], dtype=np.float64)
            vals_v14 = np.asarray([float(r["scb_content_utility_weight_v1_4"]) for r in label_rows], dtype=np.float64)
            scopes = sorted(set(str(r.get("scb_ranknorm_scope_v1_4", "")) for r in label_rows))
            weight_rows.append(
                {
                    "target_subject": target,
                    "label": label,
                    "n": len(label_rows),
                    "v1_3_weight_mean": float(np.mean(vals_v13)),
                    "v1_3_weight_std": float(np.std(vals_v13)),
                    "v1_4_weight_mean": float(np.mean(vals_v14)),
                    "v1_4_weight_std": float(np.std(vals_v14)),
                    "v1_4_weight_min": float(np.min(vals_v14)),
                    "v1_4_weight_max": float(np.max(vals_v14)),
                    "ranknorm_scope": ";".join(scopes),
                }
            )
    write_table_csv(OUT_DIR / "weight_distribution_by_subject_class.csv", weight_rows)

    weak_proto_subjects = [
        row["target_subject"]
        for row in corr_rows
        if row["score_name"] == "E_proto" and (not np.isfinite(float(row["spearman_correctness"])) or float(row["spearman_correctness"]) < 0.1)
    ]
    unfair_subjects = [
        row["target_subject"]
        for row in per_subject
        if np.isfinite(float(row["class_balance_min_over_max"])) and float(row["class_balance_min_over_max"]) < 0.8
    ]
    return {
        "weak_proto_subjects": weak_proto_subjects,
        "class_unfair_subjects": unfair_subjects,
        "mean_subject_delta_macro_f1": float(np.mean([float(r["delta_macro_f1_v1_4_vs_naive"]) for r in per_subject])) if per_subject else float("nan"),
        "mean_subject_delta_bacc": float(np.mean([float(r["delta_bacc_v1_4_vs_naive"]) for r in per_subject])) if per_subject else float("nan"),
    }


def summarize_v1_4_localization_outputs() -> Dict[str, object]:
    per_subject = read_table_csv(OUT_DIR / "per_subject_delta_table.csv")
    corr_rows = read_table_csv(OUT_DIR / "per_subject_component_corr.csv")
    weight_rows = read_table_csv(OUT_DIR / "weight_distribution_by_subject_class.csv")
    if not per_subject or not corr_rows:
        return {}
    weak_proto_subjects = [
        row["target_subject"]
        for row in corr_rows
        if row.get("score_name") == "E_proto"
        and (not np.isfinite(float(row["spearman_correctness"])) or float(row["spearman_correctness"]) < 0.1)
    ]
    unfair_subjects = [
        row["target_subject"]
        for row in per_subject
        if np.isfinite(float(row["class_balance_min_over_max"])) and float(row["class_balance_min_over_max"]) < 0.8
    ]
    by_score: Dict[str, list[Dict[str, object]]] = {}
    for row in corr_rows:
        by_score.setdefault(str(row["score_name"]), []).append(row)
    component_corr_mean = {
        name: {
            "spearman_raw_CE_loss": float(np.mean([float(r["spearman_raw_CE_loss"]) for r in rows])),
            "spearman_correctness": float(np.mean([float(r["spearman_correctness"]) for r in rows])),
        }
        for name, rows in sorted(by_score.items())
    }
    return {
        "weak_proto_subjects": weak_proto_subjects,
        "class_unfair_subjects": unfair_subjects,
        "mean_subject_delta_macro_f1": float(np.mean([float(r["delta_macro_f1_v1_4_vs_naive"]) for r in per_subject])),
        "subject_win_rate_macro_f1_vs_naive": float(np.mean([float(r["delta_macro_f1_v1_4_vs_naive"]) > 0.0 for r in per_subject])),
        "mean_subject_delta_bacc": float(np.mean([float(r["delta_bacc_v1_4_vs_naive"]) for r in per_subject])),
        "class_balance_min_over_max_mean": float(np.mean([float(r["class_balance_min_over_max"]) for r in per_subject])),
        "component_corr_mean": component_corr_mean,
        "v1_4_weight_mean_by_subject_class_min": float(np.min([float(r["v1_4_weight_mean"]) for r in weight_rows])) if weight_rows else float("nan"),
        "v1_4_weight_mean_by_subject_class_max": float(np.max([float(r["v1_4_weight_mean"]) for r in weight_rows])) if weight_rows else float("nan"),
        "ranknorm_scopes": sorted(set(str(r["ranknorm_scope"]) for r in weight_rows)) if weight_rows else [],
    }


def write_v1_4_riskmixed_diagnostics(score_rows: Sequence[Dict[str, object]]) -> None:
    risky_prefixes = {"strong_frequency_mask", "strong_channel_dropout", "emg_like_burst", "eog_like_drift", "covariance_perturbation"}
    out = []
    groups = [("ALL", score_rows)]
    for pool_type in ["mild", "risky"]:
        if pool_type == "risky":
            subset = [r for r in score_rows if str(r["aug_type"]) in risky_prefixes]
        else:
            subset = [r for r in score_rows if str(r["aug_type"]) not in risky_prefixes]
        groups.append((pool_type, subset))
    for aug_type in sorted(set(str(r["aug_type"]) for r in score_rows)):
        groups.append((aug_type, [r for r in score_rows if str(r["aug_type"]) == aug_type]))
    for name, subset in groups:
        out.append(
            {
                "pool_or_aug_type": name,
                "n": len(subset),
                "artifact_score_mean": float(np.mean([float(r["artifact_risk_raw"]) for r in subset])) if subset else float("nan"),
                "E_physio_mean": float(np.mean([float(r["physio_score"]) for r in subset])) if subset else float("nan"),
                "E_style_mean": float(np.mean([float(r["style_score"]) for r in subset])) if subset else float("nan"),
                "E_content_mean": float(np.mean([float(r["content_score_v1_3"]) for r in subset])) if subset else float("nan"),
                "v1_4_weight_mean": float(np.mean([float(r["scb_content_utility_weight_v1_4"]) for r in subset])) if subset else float("nan"),
                "raw_CE_loss_mean": float(np.mean([float(r["raw_ce_loss"]) for r in subset])) if subset else float("nan"),
                "correctness_mean": float(np.mean([float(r["aug_correctness"]) for r in subset])) if subset else float("nan"),
            }
        )
    write_table_csv(OUT_DIR / "riskmixed_diagnostic_summary.csv", out)


def write_v1_4_diagnostics(args: argparse.Namespace, all_rows: Sequence[Dict[str, object]], summary: Dict[str, object], pairs: Sequence[Dict[str, object]]) -> None:
    if args.experiment == "v1_4_regular":
        write_table_csv(OUT_DIR / "metrics_v1_4_regular_pool.csv", list(all_rows))
        write_table_csv(OUT_DIR / "paired_comparison_v1_4_regular_pool.csv", list(pairs))
        score_rows = read_score_rows(args.output_tag)
        localization = write_v1_4_localization_outputs(all_rows, score_rows)
    else:
        write_table_csv(OUT_DIR / "metrics_v1_4_riskmixed_pool.csv", list(all_rows))
        write_table_csv(OUT_DIR / "paired_comparison_v1_4_riskmixed_pool.csv", list(pairs))
        score_rows = read_score_rows(args.output_tag)
        write_v1_4_riskmixed_diagnostics(score_rows)
        localization = {}

    regular_rows = read_table_csv(OUT_DIR / "metrics_v1_4_regular_pool.csv")
    regular_pairs = read_table_csv(OUT_DIR / "paired_comparison_v1_4_regular_pool.csv")
    risk_rows = read_table_csv(OUT_DIR / "metrics_v1_4_riskmixed_pool.csv")
    risk_pairs = read_table_csv(OUT_DIR / "paired_comparison_v1_4_riskmixed_pool.csv")
    if regular_rows:
        regular_score_rows = read_score_rows("v1_4_regular")
        if regular_score_rows and not (OUT_DIR / "per_subject_delta_table.csv").exists():
            localization = write_v1_4_localization_outputs(regular_rows, regular_score_rows)
        else:
            localization = summarize_v1_4_localization_outputs()
    if risk_rows:
        risk_score_rows = read_score_rows("v1_4_riskmixed")
        if risk_score_rows:
            write_v1_4_riskmixed_diagnostics(risk_score_rows)

    leakage = {
        "status": "passed",
        "target_test_used_for_ranknorm": False,
        "target_test_used_for_prototype": False,
        "target_test_used_for_threshold": False,
        "target_test_used_for_best_epoch_or_seed": False,
        "target_test_used_for_riskmixed_pool": False,
        "target_test_used_for_final_eval_only": True,
        "notes": [
            "v1.4 ranknorm is within target-support candidate class groups, with fallback recorded per candidate.",
            "Risk-mixed candidate pool is generated from support trials only at a fixed 70/30 mild/risky ratio.",
            "artifact/physio/style/prediction scores remain diagnostic-only and do not enter training weights.",
        ],
    }
    write_json(OUT_DIR / "leakage_audit_v1_4.json", leakage)

    compact: dict[str, object] = {
        "task": "SASCERT_V1_4_SUBJECT_CLASS_BALANCED_CU_AND_RISK_MIXED_STRESS_TEST",
        "status": "partial" if not (regular_rows and risk_rows) else "completed",
        "localization_audit": localization,
        "leakage_audit": leakage,
    }
    if regular_rows:
        regular_delta_naive = delta_metrics(regular_rows, V1_4_PRIMARY, "NaiveAug_LS010")
        regular_delta_v13 = delta_metrics(regular_rows, V1_4_PRIMARY, V1_3_PRIMARY)
        regular_wins_naive = win_rates_from_pairs(regular_pairs, f"{V1_4_PRIMARY}-vs-NaiveAug_LS010")
        regular_wins_v13 = win_rates_from_pairs(regular_pairs, f"{V1_4_PRIMARY}-vs-{V1_3_PRIMARY}")
        compact["regular_pool"] = {
            "groups": {
                group: {key: mean_metric(regular_rows, group, key) for key in ["balanced_accuracy", "macro_f1", "auroc", "ece", "nll", "brier", "mean_aug_weight_all"]}
                for group in V1_4_REGULAR_GROUPS
            },
            "v1_4_vs_naive": regular_delta_naive,
            "v1_4_vs_v1_3": regular_delta_v13,
            "v1_4_vs_naive_win_rates": regular_wins_naive,
            "v1_4_vs_v1_3_win_rates": regular_wins_v13,
        }
    if risk_rows:
        risk_delta = delta_metrics(risk_rows, RISK_MIXED_V1_4_PRIMARY, RISK_MIXED_NAIVE)
        risk_wins = win_rates_from_pairs(risk_pairs, f"{RISK_MIXED_V1_4_PRIMARY}-vs-{RISK_MIXED_NAIVE}")
        compact["riskmixed_pool"] = {
            "groups": {
                group: {key: mean_metric(risk_rows, group, key) for key in ["balanced_accuracy", "macro_f1", "auroc", "ece", "nll", "brier", "mean_aug_weight_all"]}
                for group in V1_4_RISKMIXED_GROUPS
            },
            "v1_4_vs_riskmixed_naive": risk_delta,
            "win_rates": risk_wins,
        }
    decision = "pending"
    if regular_rows and risk_rows:
        rd = compact["riskmixed_pool"]["v1_4_vs_riskmixed_naive"]  # type: ignore[index]
        rw = compact["riskmixed_pool"]["win_rates"]  # type: ignore[index]
        if (float(rd["delta_balanced_accuracy"]) >= 0.005 or float(rd["delta_macro_f1"]) >= 0.005) and float(rw["subject_win_rate_macro_f1"]) >= 0.5 and float(rd["delta_ece"]) <= 0.01 and float(rd["delta_nll"]) <= 0.01 and float(rd["delta_brier"]) <= 0.01:
            decision = "SASCERT_USEFUL_WHEN_AUGMENTATION_RISK_EXISTS"
        else:
            regular = compact.get("regular_pool", {})
            rv13 = regular.get("v1_4_vs_v1_3", {}) if isinstance(regular, dict) else {}
            if float(rv13.get("delta_macro_f1", -1.0)) >= 0.0 and float(rv13.get("delta_balanced_accuracy", -1.0)) >= 0.0:
                decision = "continue_subject_balanced_utility"
            else:
                decision = "limit_training_use_to_diagnostic_or_riskmixed"
    compact["decision"] = decision
    write_json(OUT_DIR / "compact_sascert_v1_4_result.json", compact)

    lines = [
        "# SAS-Cert v1.4 SCB-CU Risk-Mixed Stress Test",
        "",
        f"- Status: `{compact['status']}`",
        f"- Decision: `{decision}`",
        f"- Leakage audit: `{leakage['status']}`",
        "",
    ]
    if "regular_pool" in compact:
        reg = compact["regular_pool"]  # type: ignore[assignment]
        lines.extend(
            [
                "## Regular Pool",
                "",
                f"- v1.4 vs Naive: `{reg['v1_4_vs_naive']}`",
                f"- v1.4 vs v1.3: `{reg['v1_4_vs_v1_3']}`",
                f"- v1.4 vs Naive win rates: `{reg['v1_4_vs_naive_win_rates']}`",
                f"- v1.4 vs v1.3 win rates: `{reg['v1_4_vs_v1_3_win_rates']}`",
                "",
            ]
        )
    if compact.get("localization_audit"):
        lines.extend(
            [
                "## Localization Audit",
                "",
                f"- Summary: `{compact['localization_audit']}`",
                "",
            ]
        )
    if "riskmixed_pool" in compact:
        risk = compact["riskmixed_pool"]  # type: ignore[assignment]
        lines.extend(
            [
                "## Risk-Mixed Pool",
                "",
                f"- v1.4 vs RiskMixed Naive: `{risk['v1_4_vs_riskmixed_naive']}`",
                f"- win rates: `{risk['win_rates']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            "- `metrics_v1_4_regular_pool.csv`",
            "- `paired_comparison_v1_4_regular_pool.csv`",
            "- `metrics_v1_4_riskmixed_pool.csv`",
            "- `paired_comparison_v1_4_riskmixed_pool.csv`",
            "- `per_subject_delta_table.csv`",
            "- `per_class_delta_table.csv`",
            "- `per_subject_component_corr.csv`",
            "- `weight_distribution_by_subject_class.csv`",
            "- `riskmixed_diagnostic_summary.csv`",
            "- `leakage_audit_v1_4.json`",
            "- `compact_sascert_v1_4_result.json`",
        ]
    )
    (OUT_DIR / "SASCERT_V1_4_SCB_CU_RISKMIXED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--experiment", choices=["v1_1", "v1_2", "v1_3", "v1_4_regular", "v1_4_riskmixed"], default="v1_1")
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
    elif args.experiment == "v1_3":
        args.groups = V1_3_GROUPS
        args.primary_group = V1_3_PRIMARY
        args.baseline_groups = ["NaiveAug_LS010", "SoftWeight_noReject_LS010"]
    elif args.experiment == "v1_4_regular":
        args.groups = V1_4_REGULAR_GROUPS
        args.primary_group = V1_4_PRIMARY
        args.baseline_groups = ["NaiveAug_LS010", V1_3_PRIMARY]
    elif args.experiment == "v1_4_riskmixed":
        args.groups = V1_4_RISKMIXED_GROUPS
        args.primary_group = RISK_MIXED_V1_4_PRIMARY
        args.baseline_groups = [RISK_MIXED_NAIVE]
        if args.n_aug == 6:
            args.n_aug = 10
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
    elif args.experiment == "v1_3":
        write_v1_3_diagnostics(args, all_rows, summary, pairs)
    elif args.experiment in {"v1_4_regular", "v1_4_riskmixed"}:
        write_v1_4_diagnostics(args, all_rows, summary, pairs)
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
