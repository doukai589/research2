from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


SFREQ = 200.0


@dataclass(frozen=True)
class AugmentedSample:
    x: np.ndarray
    y: int
    original_index: int
    aug_type: str
    intensity: float
    is_bad: bool
    aug_id: str


def augment_one(x: np.ndarray, aug_type: str, rng: np.random.Generator, intensity: float, style_bank=None) -> np.ndarray:
    if aug_type == "clean":
        return clean_aug(x, rng, intensity)
    if aug_type == "bad_content":
        return bad_content(x, rng, intensity)
    if aug_type == "bad_style":
        return bad_style(x, rng, intensity, style_bank)
    if aug_type == "bad_physio":
        return bad_physio(x, rng, intensity)
    if aug_type == "bad_artifact":
        return bad_artifact(x, rng, intensity)
    raise ValueError(f"Unknown augmentation type: {aug_type}")


def generate_augmented_set(x, y, aug_type: str, per_trial: int, rng, intensity: float, style_bank=None):
    out = []
    for i in range(x.shape[0]):
        for k in range(per_trial):
            out.append(
                AugmentedSample(
                    augment_one(x[i], aug_type, rng, intensity, style_bank).astype(np.float32),
                    int(y[i]),
                    i,
                    aug_type,
                    float(intensity),
                    aug_type.startswith("bad_"),
                    f"{aug_type}_{i:04d}_{k:02d}",
                )
            )
    return out


def generate_layer2_pool(x, y, per_trial: int, rng, intensity: float, style_bank=None):
    out = []
    bad_types = ["bad_content", "bad_style", "bad_physio", "bad_artifact"]
    clean_n = max(1, int(round(per_trial * 0.7)))
    bad_n = max(1, per_trial - clean_n)
    for i in range(x.shape[0]):
        for k in range(clean_n):
            out.append(AugmentedSample(clean_aug(x[i], rng, 0.35).astype(np.float32), int(y[i]), i, "clean", 0.35, False, f"clean_{i:04d}_{k:02d}"))
        for k in range(bad_n):
            aug_type = bad_types[int(rng.integers(0, len(bad_types)))]
            out.append(AugmentedSample(augment_one(x[i], aug_type, rng, intensity, style_bank).astype(np.float32), int(y[i]), i, aug_type, intensity, True, f"{aug_type}_{i:04d}_{k:02d}"))
    return out


def clean_aug(x, rng, intensity):
    y = x.copy()
    scale = float(np.std(y) + 1e-6)
    y += rng.normal(0.0, scale * (0.025 + 0.03 * intensity), size=y.shape)
    max_shift = max(1, int(y.shape[1] * 0.015 * (1.0 + intensity)))
    y = np.roll(y, int(rng.integers(-max_shift, max_shift + 1)), axis=1)
    if rng.random() < 0.35:
        chans = rng.choice(y.shape[0], size=max(1, int(round(y.shape[0] * 0.04))), replace=False)
        y[chans] = y[chans].mean(axis=1, keepdims=True)
    return y


def bad_content(x, rng, intensity):
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spec = np.fft.rfft(x, axis=1)
    band_idx = np.where((freqs >= 8.0) & (freqs <= 30.0))[0]
    n_mask = max(1, int(round(len(band_idx) * np.clip(intensity, 0.0, 0.9))))
    chosen = rng.choice(band_idx, size=n_mask, replace=False)
    spec[:, chosen] *= 1.0 - 0.95 * intensity
    spec[:, chosen] *= np.exp(1j * rng.uniform(-np.pi, np.pi, size=(x.shape[0], n_mask)))
    return np.fft.irfft(spec, n=x.shape[1], axis=1)


def bad_style(x, rng, intensity, style_bank):
    if style_bank is None or len(style_bank) == 0:
        return clean_aug(x, rng, 0.7)
    ref = style_bank[int(rng.integers(0, len(style_bank)))]
    x_mean, x_std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
    r_mean, r_std = ref.mean(axis=1, keepdims=True), ref.std(axis=1, keepdims=True) + 1e-6
    styled = (x - x_mean) / x_std * r_std + r_mean
    return (1.0 - intensity) * x + intensity * styled


def bad_physio(x, rng, intensity):
    y = x.copy()
    n = max(2, int(round(y.shape[0] * intensity)))
    chosen = rng.choice(y.shape[0], size=n, replace=False)
    shuffled = chosen.copy()
    rng.shuffle(shuffled)
    y[chosen] = y[shuffled]
    if intensity >= 0.5:
        y[chosen] *= rng.choice([-1.0, 1.0], size=(n, 1), p=[0.25, 0.75])
    return y


def bad_artifact(x, rng, intensity):
    y = x.copy()
    t = np.arange(y.shape[1], dtype=np.float32) / SFREQ
    scale = float(np.std(y) + 1e-6)
    drift = np.sin(2 * np.pi * rng.uniform(0.2, 1.5) * t + rng.uniform(0, 2 * np.pi))
    y[: min(4, y.shape[0])] += scale * (0.7 + 1.8 * intensity) * drift
    burst_len = max(20, int(y.shape[1] * (0.06 + 0.08 * intensity)))
    start = int(rng.integers(0, max(1, y.shape[1] - burst_len)))
    chans = rng.choice(y.shape[0], size=max(1, int(2 + 4 * intensity)), replace=False)
    y[chans, start : start + burst_len] += rng.normal(0.0, scale * (1.0 + 2.5 * intensity), size=(len(chans), burst_len))
    ch = int(rng.integers(0, y.shape[0]))
    y[ch] += rng.normal(0.0, scale * (2.0 + 3.0 * intensity), size=y.shape[1])
    return y


def band_power(x: np.ndarray, low: float, high: float) -> float:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.log(spec[..., mask].mean() + 1e-8))


def artifact_score_raw(x: np.ndarray) -> float:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    total = spec.mean() + 1e-8
    low = spec[..., (freqs >= 0.2) & (freqs < 4.0)].mean() / total
    high = spec[..., (freqs >= 35.0) & (freqs < 80.0)].mean() / total
    energy = np.mean(x**2, axis=-1)
    zmax = float(np.max(np.abs((energy - energy.mean()) / (energy.std() + 1e-8))))
    centered = x - x.mean(axis=-1, keepdims=True)
    var = np.mean(centered**2, axis=-1) + 1e-8
    kurt = np.mean(centered**4, axis=-1) / (var**2)
    return float(low + high + zmax + np.max(np.abs(kurt - 3.0)))


def augmentation_sanity(samples: Sequence[AugmentedSample]):
    if not samples:
        return {}
    x = np.stack([s.x for s in samples])
    return {
        "mean_abs": float(np.mean(np.abs(x))),
        "std": float(np.std(x)),
        "max_abs": float(np.max(np.abs(x))),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "mu_power": float(np.mean([band_power(s.x, 8, 13) for s in samples])),
        "beta_power": float(np.mean([band_power(s.x, 13, 30) for s in samples])),
        "artifact_score_mean": float(np.mean([artifact_score_raw(s.x) for s in samples])),
    }

