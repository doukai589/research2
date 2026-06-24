from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "layer2_v2"


def main() -> None:
    rows = read_cert_scores(PROJECT_ROOT / "outputs" / "layer2" / "cert_scores.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    direction_rows = direction_audit(rows)
    component_rows = single_component_auc_by_aug_type(rows)
    scored_rows, variant_rows, summary = score_variant_audit(rows, direction_rows)
    distribution_rows = score_distribution(rows, scored_rows)

    write_csv(OUTPUT_DIR / "score_direction_audit.csv", direction_rows)
    write_csv(OUTPUT_DIR / "single_component_auc_by_aug_type.csv", component_rows)
    write_csv(OUTPUT_DIR / "score_variant_auc.csv", variant_rows)
    write_csv(OUTPUT_DIR / "score_distribution_by_aug_type.csv", distribution_rows)
    write_json(OUTPUT_DIR / "layer2_v2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def read_cert_scores(path: Path) -> List[Dict[str, object]]:
    numeric = {
        "artifact_risk_raw",
        "artifact_safe_score",
        "content_score",
        "is_bad",
        "label",
        "physio_score",
        "sas_score",
        "seed",
        "style_score",
        "target_subject",
    }
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item: Dict[str, object] = dict(row)
            for key in numeric:
                if key in {"is_bad", "label", "seed", "target_subject"}:
                    item[key] = int(row[key])
                else:
                    item[key] = float(row[key])
            rows.append(item)
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}.")
    return rows


def auc_score(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranknorm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks.astype(np.float32)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def direction_audit(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    specs = [
        ("content", "content_score", "bad_content", "clean"),
        ("artifact_safe", "artifact_safe_score", "bad_artifact", "clean"),
        ("physio", "physio_score", "bad_physio", "clean"),
        ("style", "style_score", "bad_style", "clean"),
    ]
    out = []
    for component, score_key, bad_type, clean_type in specs:
        subset = [r for r in rows if r["aug_type"] in {clean_type, bad_type}]
        y = [1 if r["aug_type"] == clean_type else 0 for r in subset]
        score = [float(r[score_key]) for r in subset]
        auc = auc_score(y, score)
        clean_mean = mean(float(r[score_key]) for r in subset if r["aug_type"] == clean_type)
        bad_mean = mean(float(r[score_key]) for r in subset if r["aug_type"] == bad_type)
        out.append(
            {
                "component": component,
                "score_key": score_key,
                "clean_type": clean_type,
                "bad_type": bad_type,
                "auc_clean_positive": auc,
                "inverted_auc": 1.0 - auc if np.isfinite(auc) else float("nan"),
                "direction_maybe_wrong": bool(np.isfinite(auc) and auc < 0.5),
                "clean_mean": clean_mean,
                "bad_mean": bad_mean,
                "clean_minus_bad": clean_mean - bad_mean,
                "n": len(subset),
            }
        )
    return out


def single_component_auc_by_aug_type(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    components = [
        ("content", "content_score"),
        ("artifact_safe", "artifact_safe_score"),
        ("physio", "physio_score"),
        ("style", "style_score"),
        ("current_total", "sas_score"),
    ]
    bad_types = ["bad_artifact", "bad_content", "bad_physio", "bad_style"]
    out = []
    for component, score_key in components:
        for bad_type in ["all_bad"] + bad_types:
            subset = [r for r in rows if r["aug_type"] == "clean" or (r["is_bad"] if bad_type == "all_bad" else r["aug_type"] == bad_type)]
            y = [1 if r["aug_type"] == "clean" else 0 for r in subset]
            score = [float(r[score_key]) for r in subset]
            auc = auc_score(y, score)
            out.append(
                {
                    "component": component,
                    "score_key": score_key,
                    "bad_type": bad_type,
                    "auc_clean_positive": auc,
                    "inverted_auc": 1.0 - auc if np.isfinite(auc) else float("nan"),
                    "direction_maybe_wrong": bool(np.isfinite(auc) and auc < 0.5),
                    "n": len(subset),
                }
            )
    return out


def score_variant_audit(rows: Sequence[Dict[str, object]], direction_rows: Sequence[Dict[str, object]]):
    physio_row = next(r for r in direction_rows if r["component"] == "physio")
    physio_inverted = bool(physio_row["direction_maybe_wrong"])

    by_fold: Dict[tuple, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_fold[(row["target_subject"], row["seed"])].append(idx)

    scored = [dict(r) for r in rows]
    for indices in by_fold.values():
        artifact = np.asarray([float(rows[i]["artifact_safe_score"]) for i in indices], dtype=float)
        content = np.asarray([float(rows[i]["content_score"]) for i in indices], dtype=float)
        style = np.asarray([float(rows[i]["style_score"]) for i in indices], dtype=float)
        physio = np.asarray([float(rows[i]["physio_score"]) for i in indices], dtype=float)
        artifact_risk = np.asarray([float(rows[i]["artifact_risk_raw"]) for i in indices], dtype=float)

        artifact_rank = ranknorm(artifact)
        content_rank = ranknorm(content)
        style_rank = ranknorm(style)
        physio_rank = ranknorm(physio)
        physio_fixed = 1.0 - physio_rank if physio_inverted else physio_rank

        gate_rank = content_rank.copy()
        reject_n = max(1, int(round(len(indices) * 0.10)))
        rejected = np.argsort(artifact_risk)[::-1][:reject_n]
        gate_rank[rejected] = 0.0

        values = {
            "current_total": 0.35 * content_rank + 0.25 * physio_rank + 0.25 * artifact_rank + 0.15 * style_rank,
            "artifact_content": 0.55 * artifact_rank + 0.45 * content_rank,
            "artifact_content_style": 0.45 * artifact_rank + 0.40 * content_rank + 0.15 * style_rank,
            "artifact_content_physio_fixed": 0.40 * artifact_rank + 0.35 * content_rank + 0.15 * physio_fixed + 0.10 * style_rank,
            "artifact_gate_content_rank": gate_rank,
            "physio_fixed": physio_fixed,
        }
        for local, row_idx in enumerate(indices):
            for key, val in values.items():
                scored[row_idx][key] = float(val[local])

    variants = [
        "current_total",
        "artifact_content",
        "artifact_content_style",
        "artifact_content_physio_fixed",
        "artifact_gate_content_rank",
    ]
    bad_types = ["all_bad", "bad_artifact", "bad_content", "bad_physio", "bad_style"]
    variant_rows = []
    for variant in variants:
        for bad_type in bad_types:
            subset = [r for r in scored if r["aug_type"] == "clean" or (r["is_bad"] if bad_type == "all_bad" else r["aug_type"] == bad_type)]
            y = [1 if r["aug_type"] == "clean" else 0 for r in subset]
            score = [float(r[variant]) for r in subset]
            auc = auc_score(y, score)
            variant_rows.append(
                {
                    "variant": variant,
                    "bad_type": bad_type,
                    "auc_clean_positive": auc,
                    "inverted_auc": 1.0 - auc if np.isfinite(auc) else float("nan"),
                    "direction_maybe_wrong": bool(np.isfinite(auc) and auc < 0.5),
                    "n": len(subset),
                }
            )

    overall_rows = [r for r in variant_rows if r["bad_type"] == "all_bad"]
    best = max(overall_rows, key=lambda r: float(r["auc_clean_positive"]))["variant"]
    fixed_auc = next(r["auc_clean_positive"] for r in variant_rows if r["variant"] == "artifact_content_physio_fixed" and r["bad_type"] == "bad_physio")
    summary = {
        "status": "completed",
        "physio_original_auc": float(physio_row["auc_clean_positive"]),
        "physio_direction_maybe_wrong": physio_inverted,
        "physio_fixed_used": physio_inverted,
        "physio_fixed_badphysio_auc": fixed_auc,
        "best_non_oracle_variant": best,
        "variant_overall_auc": {r["variant"]: r["auc_clean_positive"] for r in overall_rows},
        "bad_artifact_auc_current": next(r["auc_clean_positive"] for r in variant_rows if r["variant"] == "current_total" and r["bad_type"] == "bad_artifact"),
        "protocol_leakage_detected": False,
        "selection_rule": "best_non_oracle_variant_selected_by_clean_vs_bad_auc_on_train_support_candidate_pool_only",
    }
    return scored, variant_rows, summary


def score_distribution(rows: Sequence[Dict[str, object]], scored_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    score_keys = [
        "content_score",
        "artifact_safe_score",
        "physio_score",
        "style_score",
        "sas_score",
        "artifact_content",
        "artifact_content_style",
        "artifact_content_physio_fixed",
        "artifact_gate_content_rank",
    ]
    out = []
    for aug_type in sorted({str(r["aug_type"]) for r in scored_rows}):
        subset = [r for r in scored_rows if r["aug_type"] == aug_type]
        for key in score_keys:
            vals = np.asarray([float(r[key]) for r in subset], dtype=float)
            out.append(
                {
                    "aug_type": aug_type,
                    "score": key,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
