from __future__ import annotations

import numpy as np


SFREQ = 200.0


def bandpower_features(x: np.ndarray) -> np.ndarray:
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    feats = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        feats.append(np.log(spec[..., mask].mean(axis=-1) + 1e-8))
    return np.stack(feats, axis=-1).astype(np.float32)


def covariance_batch(x: np.ndarray) -> np.ndarray:
    covs = []
    for sample in x:
        centered = sample - sample.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(1, sample.shape[1] - 1)
        covs.append((cov / (np.trace(cov) + 1e-8)).astype(np.float32))
    return np.stack(covs)


def artifact_features(x: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    total = spec.mean(axis=(-2, -1)) + 1e-8
    low = spec[..., (freqs >= 0.2) & (freqs < 4.0)].mean(axis=(-2, -1)) / total
    high = spec[..., (freqs >= 35.0) & (freqs < 80.0)].mean(axis=(-2, -1)) / total
    line_mask = (freqs >= 48.0) & (freqs <= 52.0)
    line = spec[..., line_mask].mean(axis=(-2, -1)) / total if line_mask.any() else np.zeros_like(total)
    energy = np.mean(x**2, axis=-1)
    zmax = np.max(np.abs((energy - energy.mean(axis=1, keepdims=True)) / (energy.std(axis=1, keepdims=True) + 1e-8)), axis=1)
    centered = x - x.mean(axis=-1, keepdims=True)
    var = np.mean(centered**2, axis=-1) + 1e-8
    kurt = np.mean(centered**4, axis=-1) / (var**2)
    kurt_max = np.max(np.abs(kurt - 3.0), axis=1)
    return np.stack([low, high, zmax, kurt_max, line], axis=1).astype(np.float32)


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1) + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def ranknorm(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True) if len(values) > 1 else 1.0
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks.astype(np.float32)

