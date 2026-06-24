from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .augmentations import AugmentedSample
from .features import (
    artifact_features,
    bandpower_features,
    cosine_similarity,
    covariance_batch,
    flat_bandpower,
    l2_distance,
)


@dataclass
class CertScore:
    aug_id: str
    original_index: int
    aug_type: str
    is_bad: bool
    content: float
    style: float
    physio: float
    artifact: float
    total: float


def score_candidates(
    candidates: Sequence[AugmentedSample],
    support_x: np.ndarray,
    model: Optional[torch.nn.Module],
    device: str,
    batch_size: int = 128,
) -> List[CertScore]:
    if not candidates:
        return []
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    orig_x = np.stack([support_x[c.original_index] for c in candidates]).astype(np.float32)

    content_raw = _content_score(cand_x, orig_x, model, device, batch_size)
    style_raw = _style_score(cand_x, support_x)
    physio_raw = _physio_score(cand_x, orig_x)
    artifact_raw = _artifact_risk(cand_x, support_x)

    content = rank_normalize(content_raw, higher_is_better=True)
    style = rank_normalize(style_raw, higher_is_better=True)
    physio = rank_normalize(physio_raw, higher_is_better=True)
    artifact = rank_normalize(artifact_raw, higher_is_better=True)
    total = content + style + physio - artifact

    scores = []
    for i, cand in enumerate(candidates):
        scores.append(
            CertScore(
                aug_id=cand.aug_id,
                original_index=cand.original_index,
                aug_type=cand.aug_type,
                is_bad=cand.is_bad,
                content=float(content[i]),
                style=float(style[i]),
                physio=float(physio[i]),
                artifact=float(artifact[i]),
                total=float(total[i]),
            )
        )
    return scores


def rank_normalize(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True) if len(values) > 1 else 1.0
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks.astype(np.float32)


def _content_score(
    cand_x: np.ndarray,
    orig_x: np.ndarray,
    model: Optional[torch.nn.Module],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if model is None:
        return cosine_similarity(flat_bandpower(cand_x), flat_bandpower(orig_x))
    cand_emb = _embeddings(model, cand_x, device, batch_size)
    orig_emb = _embeddings(model, orig_x, device, batch_size)
    return cosine_similarity(cand_emb, orig_emb)


def _style_score(cand_x: np.ndarray, support_x: np.ndarray) -> np.ndarray:
    cand_mean = cand_x.mean(axis=2)
    cand_std = cand_x.std(axis=2)
    supp_mean = support_x.mean(axis=2).mean(axis=0, keepdims=True)
    supp_std = support_x.std(axis=2).mean(axis=0, keepdims=True)
    cand_bp = flat_bandpower(cand_x)
    supp_bp = flat_bandpower(support_x).mean(axis=0, keepdims=True)
    stat_dist = l2_distance(np.concatenate([cand_mean, cand_std], axis=1), np.concatenate([supp_mean, supp_std], axis=1))
    bp_dist = l2_distance(cand_bp, supp_bp)
    return -(stat_dist + 0.5 * bp_dist)


def _physio_score(cand_x: np.ndarray, orig_x: np.ndarray) -> np.ndarray:
    cand_bp = bandpower_features(cand_x)[..., 2:5].reshape(cand_x.shape[0], -1)
    orig_bp = bandpower_features(orig_x)[..., 2:5].reshape(orig_x.shape[0], -1)
    bp_dist = l2_distance(cand_bp, orig_bp)
    cand_cov = covariance_batch(cand_x).reshape(cand_x.shape[0], -1)
    orig_cov = covariance_batch(orig_x).reshape(orig_x.shape[0], -1)
    cov_dist = l2_distance(cand_cov, orig_cov)
    return -(bp_dist + cov_dist)


def _artifact_risk(cand_x: np.ndarray, support_x: np.ndarray) -> np.ndarray:
    cand = artifact_features(cand_x)
    support = artifact_features(support_x)
    baseline = np.median(support, axis=0, keepdims=True)
    mad = np.median(np.abs(support - baseline), axis=0, keepdims=True) + 1e-6
    z = np.maximum(0.0, (cand - baseline) / mad)
    return z.mean(axis=1)


def _embeddings(model: torch.nn.Module, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            _, features = model(batch, return_features=True)
            out.append(features.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def scores_to_rows(scores: Iterable[CertScore]) -> List[Dict[str, object]]:
    rows = []
    for score in scores:
        rows.append(
            {
                "aug_id": score.aug_id,
                "original_index": score.original_index,
                "aug_type": score.aug_type,
                "is_bad": int(score.is_bad),
                "content": score.content,
                "style": score.style,
                "physio": score.physio,
                "artifact": score.artifact,
                "total": score.total,
            }
        )
    return rows
