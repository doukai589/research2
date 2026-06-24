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


def augment_one(
    x: np.ndarray,
    aug_type: str,
    rng: np.random.Generator,
    intensity: float = 0.5,
    style_bank: Optional[np.ndarray] = None,
) -> np.ndarray:
    if aug_type == "clean":
        return clean_augmentation(x, rng, intensity)
    if aug_type == "bad_content":
        return content_drift(x, rng, intensity)
    if aug_type == "bad_style":
        return style_mismatch(x, rng, intensity, style_bank)
    if aug_type == "bad_physio":
        return physiological_violation(x, rng, intensity)
    if aug_type == "bad_artifact":
        return artifact_contamination(x, rng, intensity)
    raise ValueError(f"Unknown augmentation type: {aug_type}")


def generate_augmented_set(
    x: np.ndarray,
    y: np.ndarray,
    aug_type: str,
    per_trial: int,
    rng: np.random.Generator,
    intensity: float,
    style_bank: Optional[np.ndarray] = None,
) -> Sequence[AugmentedSample]:
    out = []
    for i in range(x.shape[0]):
        for k in range(per_trial):
            aug_x = augment_one(x[i], aug_type, rng, intensity, style_bank)
            out.append(
                AugmentedSample(
                    x=aug_x.astype(np.float32),
                    y=int(y[i]),
                    original_index=i,
                    aug_type=aug_type,
                    intensity=float(intensity),
                    is_bad=aug_type.startswith("bad_"),
                    aug_id=f"{aug_type}_{i:04d}_{k:02d}",
                )
            )
    return out


def generate_mixed_candidate_pool(
    x: np.ndarray,
    y: np.ndarray,
    per_trial: int,
    rng: np.random.Generator,
    style_bank: Optional[np.ndarray] = None,
    intensity: float = 0.5,
) -> Sequence[AugmentedSample]:
    """Generate about 70% clean/mild and 30% bad/stress candidates."""

    bad_types = ["bad_content", "bad_style", "bad_physio", "bad_artifact"]
    out = []
    clean_count = max(1, int(round(per_trial * 0.7)))
    bad_count = max(1, per_trial - clean_count)
    for i in range(x.shape[0]):
        for k in range(clean_count):
            aug_x = clean_augmentation(x[i], rng, intensity=0.35)
            out.append(
                AugmentedSample(aug_x.astype(np.float32), int(y[i]), i, "clean", 0.35, False, f"clean_{i:04d}_{k:02d}")
            )
        for k in range(bad_count):
            aug_type = bad_types[int(rng.integers(0, len(bad_types)))]
            aug_x = augment_one(x[i], aug_type, rng, intensity=intensity, style_bank=style_bank)
            out.append(
                AugmentedSample(aug_x.astype(np.float32), int(y[i]), i, aug_type, intensity, True, f"{aug_type}_{i:04d}_{k:02d}")
            )
    return out


def clean_augmentation(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.35) -> np.ndarray:
    y = x.copy()
    # Mild Gaussian noise around 20-25 dB SNR.
    signal_std = float(np.std(y) + 1e-6)
    noise_scale = signal_std * (0.025 + 0.03 * intensity)
    y = y + rng.normal(0.0, noise_scale, size=y.shape)
    # Mild temporal shift.
    max_shift = max(1, int(y.shape[1] * 0.015 * (1.0 + intensity)))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    y = np.roll(y, shift, axis=1)
    # Mild channel dropout with mean replacement.
    if rng.random() < 0.35:
        n_drop = max(1, int(round(y.shape[0] * 0.04)))
        chans = rng.choice(y.shape[0], size=n_drop, replace=False)
        y[chans] = y[chans].mean(axis=1, keepdims=True)
    return y.astype(np.float32)


def content_drift(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.5) -> np.ndarray:
    """Mask or phase-disrupt MI-relevant mu/beta bands."""

    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spectrum = np.fft.rfft(x, axis=1)
    mi_band = (freqs >= 8.0) & (freqs <= 30.0)
    mask_ratio = np.clip(intensity, 0.0, 0.9)
    if mi_band.any():
        band_idx = np.where(mi_band)[0]
        n_mask = max(1, int(round(len(band_idx) * mask_ratio)))
        chosen = rng.choice(band_idx, size=n_mask, replace=False)
        spectrum[:, chosen] *= 1.0 - 0.95 * intensity
        phase_noise = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(x.shape[0], n_mask)))
        spectrum[:, chosen] *= phase_noise
    return np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(np.float32)


def style_mismatch(
    x: np.ndarray,
    rng: np.random.Generator,
    intensity: float = 0.5,
    style_bank: Optional[np.ndarray] = None,
) -> np.ndarray:
    if style_bank is None or len(style_bank) == 0:
        return clean_augmentation(x, rng, intensity=0.7)
    ref = style_bank[int(rng.integers(0, len(style_bank)))]
    x_mean = x.mean(axis=1, keepdims=True)
    x_std = x.std(axis=1, keepdims=True) + 1e-6
    ref_mean = ref.mean(axis=1, keepdims=True)
    ref_std = ref.std(axis=1, keepdims=True) + 1e-6
    normalized = (x - x_mean) / x_std
    styled = normalized * ref_std + ref_mean
    lam = float(np.clip(intensity, 0.0, 1.0))
    return ((1.0 - lam) * x + lam * styled).astype(np.float32)


def physiological_violation(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.5) -> np.ndarray:
    y = x.copy()
    n_chans = y.shape[0]
    n_perm = max(2, int(round(n_chans * np.clip(intensity, 0.0, 1.0))))
    chosen = rng.choice(n_chans, size=n_perm, replace=False)
    shuffled = chosen.copy()
    rng.shuffle(shuffled)
    y[chosen] = y[shuffled]
    if intensity >= 0.5:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, 1), p=[0.25, 0.75])
        y[chosen] *= signs
    return y.astype(np.float32)


def artifact_contamination(x: np.ndarray, rng: np.random.Generator, intensity: float = 0.5) -> np.ndarray:
    y = x.copy()
    t = np.arange(y.shape[1], dtype=np.float32) / SFREQ
    signal_std = float(np.std(y) + 1e-6)
    # EOG-like slow drift.
    drift_freq = rng.uniform(0.2, 1.5)
    drift = np.sin(2 * np.pi * drift_freq * t + rng.uniform(0, 2 * np.pi))
    drift_amp = signal_std * (0.7 + 1.8 * intensity)
    frontal = np.arange(min(4, y.shape[0]))
    y[frontal] += drift_amp * drift
    # EMG-like high-frequency burst.
    burst_len = max(20, int(y.shape[1] * (0.06 + 0.08 * intensity)))
    start = int(rng.integers(0, max(1, y.shape[1] - burst_len)))
    chans = rng.choice(y.shape[0], size=max(1, int(2 + 4 * intensity)), replace=False)
    burst = rng.normal(0.0, signal_std * (1.0 + 2.5 * intensity), size=(len(chans), burst_len))
    y[chans, start : start + burst_len] += burst
    # One bad channel.
    if rng.random() < 0.5 + 0.4 * intensity:
        ch = int(rng.integers(0, y.shape[0]))
        y[ch] += rng.normal(0.0, signal_std * (2.0 + 3.0 * intensity), size=y.shape[1])
    return y.astype(np.float32)

