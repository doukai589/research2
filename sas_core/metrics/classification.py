"""Classification and calibration metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.zeros_like(probs, dtype=np.float32)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    try:
        auroc = float(roc_auc_score(y_true, probs[:, 1])) if probs.shape[1] == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, pred)),
        "auroc": auroc,
        "nll": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
        "ece": ece_score(y_true, probs),
        "brier": brier_score(y_true, probs),
    }

