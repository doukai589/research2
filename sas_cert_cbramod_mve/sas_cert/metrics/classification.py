from __future__ import annotations

from typing import Dict

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def classification_metrics(logits: np.ndarray, y_true: np.ndarray, n_classes: int = 4) -> Dict[str, float]:
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    return {
        "acc": float(np.mean(y_pred == y_true)),
        "macro_f1": float(macro_f1(y_true, y_pred, n_classes)),
        "kappa": float(cohen_kappa(y_true, y_pred, n_classes)),
        "nll": float(-np.mean(np.log(np.maximum(probs[np.arange(len(y_true)), y_true], 1e-12)))),
        "ece": float(expected_calibration_error(probs, y_true)),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    vals = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        vals.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(vals))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total * total)
    return 0.0 if abs(1.0 - pe) < 1e-12 else float((po - pe) / (1.0 - pe))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = rankdata(scores)
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)

