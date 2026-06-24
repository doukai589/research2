from __future__ import annotations

import numpy as np


SFREQ = 200.0


def bandpower_features(x: np.ndarray) -> np.ndarray:
    """Return per-channel bandpower for delta/theta/alpha/mu/beta/gamma-lite."""

    bands = [(1, 4), (4, 8), (8, 12), (12, 16), (16, 30), (30, 45)]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    feats = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            feats.append(np.log(spec[..., mask].mean(axis=-1) + 1e-8))
        else:
            feats.append(np.zeros(x.shape[:-1], dtype=np.float32))
    return np.stack(feats, axis=-1).astype(np.float32)


def flat_bandpower(x: np.ndarray) -> np.ndarray:
    return bandpower_features(x).reshape(x.shape[0], -1).astype(np.float32)


def covariance_matrix(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(1, x.shape[1] - 1)
    trace = float(np.trace(cov) + 1e-8)
    return (cov / trace).astype(np.float32)


def covariance_batch(x: np.ndarray) -> np.ndarray:
    return np.stack([covariance_matrix(sample) for sample in x]).astype(np.float32)


def artifact_features(x: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    total = spec.mean(axis=(-2, -1)) + 1e-8
    low = spec[..., (freqs >= 0.2) & (freqs < 4.0)].mean(axis=(-2, -1)) / total
    high = spec[..., (freqs >= 35.0) & (freqs < 80.0)].mean(axis=(-2, -1)) / total
    channel_energy = np.mean(x**2, axis=-1)
    energy_zmax = np.max(np.abs(_zscore(channel_energy, axis=1)), axis=1)
    centered = x - x.mean(axis=-1, keepdims=True)
    var = np.mean(centered**2, axis=-1) + 1e-8
    kurt = np.mean(centered**4, axis=-1) / (var**2)
    kurt_max = np.max(np.abs(kurt - 3.0), axis=1)
    return np.stack([low, high, energy_zmax, kurt_max], axis=1).astype(np.float32)


def _zscore(x: np.ndarray, axis: int) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-8)


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return np.sum(a * b, axis=1) / denom

