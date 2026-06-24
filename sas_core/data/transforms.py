"""Array transforms shared by EEG experiments."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample


def chunked_resample(X: np.ndarray, target_len: int, chunk: int = 512) -> np.ndarray:
    if X.shape[-1] == target_len:
        return X.astype(np.float32, copy=False)
    out = np.empty((X.shape[0], X.shape[1], target_len), dtype=np.float32)
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        out[start:stop] = resample(X[start:stop], target_len, axis=-1).astype(np.float32)
    return out


def to_steegformer_input(X160: np.ndarray) -> np.ndarray:
    """Convert canonical 160 Hz, 4 s PhysioNetMI windows to 128 Hz ST input."""

    return chunked_resample(X160, 512)

