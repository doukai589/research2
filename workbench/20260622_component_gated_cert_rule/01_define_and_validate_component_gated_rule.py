#!/usr/bin/env python3
"""Define and score-validate component-gated SAS-Cert rule v1."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = [90, 91, 92]
SEEDS = [20, 21]
SOURCES = {
    "ST-EEGFormer-small_source_tuned": ROOT / "workbench" / "20260622_steegformer_sascert_core" / "outputs" / "score_rows" / "st_source_tuned_full",
    "CBraMod_frozen": ROOT / "workbench" / "20260622_cbramod_physionetmi_sascert_matched" / "outputs" / "score_rows" / "mini",
}
COMPONENTS = [
    "sas_score",
    "physio_score",
    "style_score",
    "artifact_safe_score",
    "content_score",
    "score_artifact_gate_physio",
    "component_gated_v1",
    "component_gated_v1_soft",
]


def ranknorm(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(arr), endpoint=True)
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks


def read_rows(backbone: str, score_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            path = score_dir / f"target{target}_seed{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    item: dict[str, object] = dict(row)
                    item["backbone"] = backbone
                    item["target_subject"] = target
                    item["seed"] = seed
                    item["fold"] = f"target{target}_seed{seed}"
                    item["is_clean"] = int(row["aug_type"] == "clean")
                    for key in ["content_score", "style_score", "physio_score", "artifact_safe_score", "artifact_risk_raw", "sas_score"]:
                        item[key] = float(row[key])
                    rows.append(item)
    return rows


def add_component_gated_scores(rows: list[dict[str, object]]) -> None:
    by_fold: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_fold.setdefault((str(row["backbone"]), str(row["fold"])), []).append(row)
    for fold_rows in by_fold.values():
        artifact_risk = np.asarray([float(row["artifact_risk_raw"]) for row in fold_rows], dtype=np.float64)
        artifact_threshold = float(np.percentile(artifact_risk, 90.0))
        base = 0.75 * np.asarray([float(row["physio_score"]) for row in fold_rows]) + 0.25 * np.asarray(
            [float(row["style_score"]) for row in fold_rows]
        )
        base_rank = ranknorm(base, True)
        soft_artifact_gate = ranknorm(artifact_risk, False)
        for idx, row in enumerate(fold_rows):
            hard_pass = float(float(row["artifact_risk_raw"]) < artifact_threshold)
            row["artifact_gate_pass"] = int(hard_pass > 0.0)
            row["component_gated_v1"] = float(base_rank[idx] * hard_pass)
            row["component_gated_v1_soft"] = float(base_rank[idx] * (0.2 + 0.8 * soft_artifact_gate[idx]))
            # Content is retained as a diagnostic warning, not as a positive term.
            row["content_unstable_warning"] = int(float(row["content_score"]) >= 0.75 or float(row["content_score"]) <= 0.25)
            row["score_artifact_gate_physio"] = 0.0 if hard_pass == 0.0 else float(row["physio_score"])


def auc_or_none(y: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64)))


def component_auc(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        subset = [row for row in rows if row["backbone"] == backbone]
        y = [int(row["is_clean"]) for row in subset]
        for component in COMPONENTS:
            auc = auc_or_none(y, [float(row[component]) for row in subset])
            out.append(
                {
                    "backbone": backbone,
                    "component": component,
                    "comparison": "clean_vs_all_bad",
                    "auc_high_score_is_clean": auc,
                    "n": len(subset),
                }
            )
    return out


def bad_type_auc(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        for bad_type in ["bad_artifact", "bad_content", "bad_physio"]:
            subset = [row for row in rows if row["backbone"] == backbone and row["aug_type"] in {"clean", bad_type}]
            y = [int(row["is_clean"]) for row in subset]
            for component in COMPONENTS:
                auc = auc_or_none(y, [float(row[component]) for row in subset])
                out.append(
                    {
                        "backbone": backbone,
                        "bad_type": bad_type,
                        "component": component,
                        "auc_high_score_is_clean": auc,
                        "n": len(subset),
                    }
                )
    return out


def gate_stats(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        for aug_type in sorted(set(str(row["aug_type"]) for row in rows if row["backbone"] == backbone)):
            subset = [row for row in rows if row["backbone"] == backbone and row["aug_type"] == aug_type]
            out.append(
                {
                    "backbone": backbone,
                    "aug_type": aug_type,
                    "artifact_gate_reject_rate": float(np.mean([1 - int(row["artifact_gate_pass"]) for row in subset])),
                    "content_warning_rate": float(np.mean([int(row["content_unstable_warning"]) for row in subset])),
                    "n": len(subset),
                }
            )
    return out


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_by_backbone(component_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in component_rows)):
        rows = [row for row in component_rows if row["backbone"] == backbone]
        best = max(rows, key=lambda row: float(row["auc_high_score_is_clean"]))
        gated = next(row for row in rows if row["component"] == "component_gated_v1")
        current = next(row for row in rows if row["component"] == "sas_score")
        out.append(
            {
                "backbone": backbone,
                "best_component": best["component"],
                "best_auc": best["auc_high_score_is_clean"],
                "component_gated_v1_auc": gated["auc_high_score_is_clean"],
                "current_sas_auc": current["auc_high_score_is_clean"],
            }
        )
    return out


def make_report(payload: dict[str, object]) -> str:
    lines = [
        "# Component-Gated Certificate Rule v1",
        "",
        "## Rule",
        "",
        "```text",
        "artifact_gate_pass = artifact_risk < fold_p90",
        "base = 0.75 * physio_score + 0.25 * style_score",
        "component_gated_v1 = ranknorm(base) * artifact_gate_pass",
        "component_gated_v1_soft = ranknorm(base) * (0.2 + 0.8 * artifact_safe_rank)",
        "content_score is diagnostic only and is not a positive term in v1",
        "```",
        "",
        "## Score-Only AUC",
        "",
        "| Backbone | Current SAS | ArtifactGatePhysio | ComponentGated v1 | ComponentGated v1 soft | Best |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rows_by_backbone: dict[str, dict[str, float]] = {}
    for row in payload["component_auc"]:
        rows_by_backbone.setdefault(str(row["backbone"]), {})[str(row["component"])] = float(row["auc_high_score_is_clean"])
    best_map = {row["backbone"]: row for row in payload["best_by_backbone"]}
    for backbone in sorted(rows_by_backbone):
        vals = rows_by_backbone[backbone]
        best = best_map[backbone]
        lines.append(
            f"| {backbone} | {vals['sas_score']:.4f} | {vals['score_artifact_gate_physio']:.4f} | {vals['component_gated_v1']:.4f} | {vals['component_gated_v1_soft']:.4f} | {best['best_component']} `{float(best['best_auc']):.4f}` |"
        )
    lines.extend(["", "## Gate Stats", "", "| Backbone | Aug type | Artifact reject rate | Content warning rate |", "|---|---|---:|---:|"])
    for row in payload["gate_stats"]:
        lines.append(
            f"| {row['backbone']} | {row['aug_type']} | {float(row['artifact_gate_reject_rate']):.4f} | {float(row['content_warning_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "`component_gated_v1_defined_score_validated`",
            "",
            "The v1 rule fixes the obvious scalar-score direction failure, but it does not beat the simpler `score_artifact_gate_physio` diagnostic score. Use v1 as the interpretable certificate policy, and treat `score_artifact_gate_physio` as the current strongest score-only baseline for the next small ST reliability test.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for backbone, score_dir in SOURCES.items():
        rows.extend(read_rows(backbone, score_dir))
    add_component_gated_scores(rows)
    component_rows = component_auc(rows)
    bad_rows = bad_type_auc(rows)
    gate_rows = gate_stats(rows)
    best_rows = best_by_backbone(component_rows)
    payload = {
        "status": "completed",
        "rule": {
            "name": "component_gated_v1",
            "artifact_gate": "artifact_risk < fold_p90",
            "base_score": "0.75 * physio_score + 0.25 * style_score",
            "content_policy": "diagnostic_only_not_positive_term",
            "uses_existing_score_rows_only": True,
            "no_retraining": True,
            "no_target_test_labels_used": True,
        },
        "component_auc": component_rows,
        "bad_type_auc": bad_rows,
        "gate_stats": gate_rows,
        "best_by_backbone": best_rows,
        "decision": "component_gated_v1_defined_score_validated",
        "next_step": "test component_gated_v1 or artifact_gate_physio in a small ST reliability setting before any full expansion",
    }
    write_csv(OUT_DIR / "component_gated_v1_component_auc.csv", component_rows)
    write_csv(OUT_DIR / "component_gated_v1_bad_type_auc.csv", bad_rows)
    write_csv(OUT_DIR / "component_gated_v1_gate_stats.csv", gate_rows)
    write_csv(OUT_DIR / "component_gated_v1_best_by_backbone.csv", best_rows)
    (OUT_DIR / "compact_component_gated_cert_rule_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "COMPONENT_GATED_CERT_RULE_V1.md").write_text(make_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
