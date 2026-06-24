from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from sas_cert.augmentations import AugmentedSample
from sas_cert.cert.features import artifact_features, bandpower_features, cosine, covariance_batch, l2, ranknorm


def score_candidates(
    candidates: Sequence[AugmentedSample],
    support_x: np.ndarray,
    support_features: np.ndarray,
    candidate_features: np.ndarray,
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    if weights is None:
        weights = {"content": 0.35, "physio": 0.25, "artifact_safe": 0.25, "style": 0.15}
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    orig_x = np.stack([support_x[c.original_index] for c in candidates]).astype(np.float32)
    orig_feat = np.stack([support_features[c.original_index] for c in candidates]).astype(np.float32)

    content_raw = cosine(candidate_features, orig_feat)
    style_raw = style_score(cand_x, support_x)
    physio_raw = physio_score(cand_x, orig_x)
    artifact_risk = artifact_risk_score(cand_x, support_x)

    content = ranknorm(content_raw, True)
    style = ranknorm(style_raw, True)
    physio = ranknorm(physio_raw, True)
    artifact_safe = ranknorm(artifact_risk, False)
    total = (
        weights["content"] * content
        + weights["physio"] * physio
        + weights["artifact_safe"] * artifact_safe
        + weights["style"] * style
    )
    rows = []
    for i, cand in enumerate(candidates):
        rows.append(
            {
                "aug_id": cand.aug_id,
                "sample_id": cand.aug_id,
                "original_id": int(cand.original_index),
                "aug_type": cand.aug_type,
                "is_bad": int(cand.is_bad),
                "label": int(cand.y),
                "content_score": float(content[i]),
                "style_score": float(style[i]),
                "physio_score": float(physio[i]),
                "artifact_safe_score": float(artifact_safe[i]),
                "artifact_risk_raw": float(artifact_risk[i]),
                "sas_score": float(total[i]),
            }
        )
    if any(not np.isfinite(row["sas_score"]) for row in rows):
        raise RuntimeError("SAS score contains NaN/Inf.")
    return rows


def style_score(cand_x: np.ndarray, support_x: np.ndarray) -> np.ndarray:
    cand_stats = np.concatenate([cand_x.mean(axis=2), cand_x.std(axis=2)], axis=1)
    anchor_stats = np.concatenate([support_x.mean(axis=2).mean(axis=0), support_x.std(axis=2).mean(axis=0)])[None, :]
    cand_bp = bandpower_features(cand_x)[..., 2:4].reshape(cand_x.shape[0], -1)
    anchor_bp = bandpower_features(support_x)[..., 2:4].reshape(support_x.shape[0], -1).mean(axis=0, keepdims=True)
    return -(l2(cand_stats, anchor_stats) + 0.5 * l2(cand_bp, anchor_bp))


def physio_score(cand_x: np.ndarray, orig_x: np.ndarray) -> np.ndarray:
    cand_bp = bandpower_features(cand_x)[..., 2:4].reshape(cand_x.shape[0], -1)
    orig_bp = bandpower_features(orig_x)[..., 2:4].reshape(orig_x.shape[0], -1)
    cand_cov = covariance_batch(cand_x).reshape(cand_x.shape[0], -1)
    orig_cov = covariance_batch(orig_x).reshape(orig_x.shape[0], -1)
    return -(l2(cand_bp, orig_bp) + l2(cand_cov, orig_cov))


def artifact_risk_score(cand_x: np.ndarray, support_x: np.ndarray) -> np.ndarray:
    cand = artifact_features(cand_x)
    support = artifact_features(support_x)
    median = np.median(support, axis=0, keepdims=True)
    mad = np.median(np.abs(support - median), axis=0, keepdims=True) + 1e-6
    return np.maximum(0.0, (cand - median) / mad).mean(axis=1)

