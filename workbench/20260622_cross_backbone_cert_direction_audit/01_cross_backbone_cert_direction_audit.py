#!/usr/bin/env python3
"""Cross-backbone certificate direction audit for ST-EEGFormer and CBraMod.

This audit uses existing score rows only. It does not retrain models, regenerate
augmentations, or use held-out target labels.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = [90, 91, 92]
SEEDS = [20, 21]

SOURCES = {
    "ST-EEGFormer-small_source_tuned": {
        "score_dir": ROOT / "workbench" / "20260622_steegformer_sascert_core" / "outputs" / "score_rows" / "st_source_tuned_full",
        "note": "source-tuned ST feature extractor, final target score rows",
    },
    "CBraMod_frozen": {
        "score_dir": ROOT / "workbench" / "20260622_cbramod_physionetmi_sascert_matched" / "outputs" / "score_rows" / "mini",
        "note": "frozen CBraMod mini matrix score rows",
    },
}

BASE_COMPONENTS = ["content_score", "style_score", "physio_score", "artifact_safe_score", "sas_score"]
BAD_TYPES = ["bad_artifact", "bad_content", "bad_physio"]


def read_rows(backbone: str, score_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            path = score_dir / f"target{target}_seed{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing score rows for {backbone}: {path}")
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item: dict[str, object] = dict(row)
                    item["backbone"] = backbone
                    item["target_subject"] = target
                    item["seed"] = seed
                    item["fold"] = f"target{target}_seed{seed}"
                    item["is_clean"] = int(row["aug_type"] == "clean")
                    item["is_bad"] = int(row["aug_type"] != "clean")
                    for key in BASE_COMPONENTS + ["artifact_risk_raw"]:
                        item[key] = float(row[key])
                    rows.append(item)
    return rows


def auc_or_none(y: Sequence[int], score: Sequence[float]) -> float | None:
    y_arr = np.asarray(y, dtype=np.int64)
    if len(set(y_arr.tolist())) < 2:
        return None
    return float(roc_auc_score(y_arr, np.asarray(score, dtype=np.float64)))


def add_derived_scores(rows: list[dict[str, object]]) -> None:
    by_fold: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_fold.setdefault((str(row["backbone"]), str(row["fold"])), []).append(row)
    for fold_rows in by_fold.values():
        artifact_risk = np.asarray([float(row["artifact_risk_raw"]) for row in fold_rows], dtype=np.float64)
        threshold = float(np.percentile(artifact_risk, 90.0))
        for row in fold_rows:
            physio = float(row["physio_score"])
            row["score_artifact_gate_physio"] = 0.0 if float(row["artifact_risk_raw"]) >= threshold else physio


def component_auc(rows: Sequence[dict[str, object]], components: Sequence[str]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        subset = [row for row in rows if row["backbone"] == backbone]
        y = [int(row["is_clean"]) for row in subset]
        for component in components:
            auc = auc_or_none(y, [float(row[component]) for row in subset])
            out.append(
                {
                    "backbone": backbone,
                    "component": component,
                    "comparison": "clean_vs_all_bad",
                    "auc_high_score_is_clean": auc,
                    "direction": direction_label(auc),
                    "n": len(subset),
                }
            )
    return out


def bad_type_auc(rows: Sequence[dict[str, object]], components: Sequence[str]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        for bad_type in BAD_TYPES:
            subset = [row for row in rows if row["backbone"] == backbone and row["aug_type"] in {"clean", bad_type}]
            y = [int(row["is_clean"]) for row in subset]
            for component in components:
                auc = auc_or_none(y, [float(row[component]) for row in subset])
                out.append(
                    {
                        "backbone": backbone,
                        "bad_type": bad_type,
                        "component": component,
                        "auc_high_score_is_clean": auc,
                        "direction": direction_label(auc),
                        "n": len(subset),
                    }
                )
    return out


def direction_label(auc: float | None) -> str:
    if auc is None:
        return "undefined"
    if auc >= 0.6:
        return "clean_high"
    if auc <= 0.4:
        return "bad_high_or_inverted"
    return "weak_or_mixed"


def score_distributions(rows: Sequence[dict[str, object]], components: Sequence[str]) -> list[dict[str, object]]:
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        for aug_type in sorted(set(str(row["aug_type"]) for row in rows if row["backbone"] == backbone)):
            subset = [row for row in rows if row["backbone"] == backbone and row["aug_type"] == aug_type]
            for component in components:
                values = np.asarray([float(row[component]) for row in subset], dtype=np.float64)
                out.append(
                    {
                        "backbone": backbone,
                        "aug_type": aug_type,
                        "component": component,
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "n": len(values),
                    }
                )
    return out


def direction_conflicts(component_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_component: dict[str, dict[str, dict[str, object]]] = {}
    for row in component_rows:
        by_component.setdefault(str(row["component"]), {})[str(row["backbone"])] = dict(row)
    out = []
    for component, by_backbone in sorted(by_component.items()):
        if len(by_backbone) < 2:
            continue
        names = sorted(by_backbone)
        a, b = by_backbone[names[0]], by_backbone[names[1]]
        auc_a = float(a["auc_high_score_is_clean"])
        auc_b = float(b["auc_high_score_is_clean"])
        dir_a = str(a["direction"])
        dir_b = str(b["direction"])
        conflict = bool(dir_a != dir_b and "weak" not in dir_a and "weak" not in dir_b)
        out.append(
            {
                "component": component,
                "backbone_a": names[0],
                "auc_a": auc_a,
                "direction_a": dir_a,
                "backbone_b": names[1],
                "auc_b": auc_b,
                "direction_b": dir_b,
                "abs_auc_gap": abs(auc_a - auc_b),
                "direction_conflict": conflict,
            }
        )
    return out


def bad_type_direction_conflicts(bad_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in bad_rows:
        by_key.setdefault((str(row["bad_type"]), str(row["component"])), {})[str(row["backbone"])] = dict(row)
    out = []
    for (bad_type, component), by_backbone in sorted(by_key.items()):
        if len(by_backbone) < 2:
            continue
        names = sorted(by_backbone)
        a, b = by_backbone[names[0]], by_backbone[names[1]]
        auc_a = float(a["auc_high_score_is_clean"])
        auc_b = float(b["auc_high_score_is_clean"])
        dir_a = str(a["direction"])
        dir_b = str(b["direction"])
        conflict = bool(dir_a != dir_b and "weak" not in dir_a and "weak" not in dir_b)
        out.append(
            {
                "bad_type": bad_type,
                "component": component,
                "backbone_a": names[0],
                "auc_a": auc_a,
                "direction_a": dir_a,
                "backbone_b": names[1],
                "auc_b": auc_b,
                "direction_b": dir_b,
                "abs_auc_gap": abs(auc_a - auc_b),
                "direction_conflict": conflict,
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_variants(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    variants = [
        "content_score",
        "style_score",
        "physio_score",
        "artifact_safe_score",
        "sas_score",
        "score_artifact_gate_physio",
    ]
    out = []
    for backbone in sorted(set(str(row["backbone"]) for row in rows)):
        subset = [row for row in rows if row["backbone"] == backbone]
        y = [int(row["is_clean"]) for row in subset]
        scored = []
        for variant in variants:
            auc = auc_or_none(y, [float(row[variant]) for row in subset])
            scored.append((variant, auc if auc is not None else float("nan")))
        best = max(scored, key=lambda x: x[1])
        current = next(x for x in scored if x[0] == "sas_score")
        out.append(
            {
                "backbone": backbone,
                "best_variant": best[0],
                "best_auc": best[1],
                "current_sas_auc": current[1],
                "current_sas_direction": direction_label(current[1]),
            }
        )
    return out


def make_report(payload: dict[str, object]) -> str:
    component_rows = payload["component_auc_by_backbone"]
    conflicts = payload["direction_conflict_table"]
    best = payload["best_variants"]
    lines = [
        "# Cross-Backbone Certificate Direction Audit",
        "",
        "## Scope",
        "",
        "- Dataset: `PhysioNetMI`",
        "- Targets: `90,91,92`",
        "- Seeds: `20,21`",
        "- Backbones: `ST-EEGFormer-small_source_tuned`, `CBraMod_frozen`",
        "- Inputs: existing score rows only; no retraining and no new augmentation generation.",
        "",
        "## Component AUC",
        "",
        "| Backbone | Component | AUC high=clean | Direction |",
        "|---|---|---:|---|",
    ]
    for row in component_rows:
        lines.append(
            f"| {row['backbone']} | {row['component']} | {float(row['auc_high_score_is_clean']):.4f} | {row['direction']} |"
        )
    lines.extend(["", "## Direction Conflicts", "", "| Component | ST AUC/Dir | CBraMod AUC/Dir | Conflict |", "|---|---:|---:|---|"])
    for row in conflicts:
        st_side = row if "ST-EEGFormer" in str(row["backbone_a"]) else None
        if st_side is None:
            st_auc = row["auc_b"]
            st_dir = row["direction_b"]
            cb_auc = row["auc_a"]
            cb_dir = row["direction_a"]
        else:
            st_auc = row["auc_a"]
            st_dir = row["direction_a"]
            cb_auc = row["auc_b"]
            cb_dir = row["direction_b"]
        lines.append(f"| {row['component']} | {float(st_auc):.4f} / {st_dir} | {float(cb_auc):.4f} / {cb_dir} | {row['direction_conflict']} |")
    lines.extend(["", "## Bad-Type Direction Conflicts", "", "| Bad type | Component | ST AUC/Dir | CBraMod AUC/Dir | Conflict |", "|---|---|---:|---:|---|"])
    for row in payload["bad_type_direction_conflict_table"]:
        st_side = row if "ST-EEGFormer" in str(row["backbone_a"]) else None
        if st_side is None:
            st_auc = row["auc_b"]
            st_dir = row["direction_b"]
            cb_auc = row["auc_a"]
            cb_dir = row["direction_a"]
        else:
            st_auc = row["auc_a"]
            st_dir = row["direction_a"]
            cb_auc = row["auc_b"]
            cb_dir = row["direction_b"]
        if row["direction_conflict"] or float(row["abs_auc_gap"]) >= 0.3:
            lines.append(
                f"| {row['bad_type']} | {row['component']} | {float(st_auc):.4f} / {st_dir} | {float(cb_auc):.4f} / {cb_dir} | {row['direction_conflict']} |"
            )
    lines.extend(["", "## Best Variants", "", "| Backbone | Best variant | Best AUC | Current SAS AUC | Current SAS direction |", "|---|---|---:|---:|---|"])
    for row in best:
        lines.append(
            f"| {row['backbone']} | {row['best_variant']} | {float(row['best_auc']):.4f} | {float(row['current_sas_auc']):.4f} | {row['current_sas_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The existing total `sas_score` is directionally wrong on both backbones in the matched PhysioNetMI mini scope.",
            "- The most useful score-only variant on both backbones is `score_artifact_gate_physio`.",
            "- The strongest component-level backbone difference is `content_score` on `bad_content`: CBraMod treats clean as high, while ST treats bad content as high.",
            "- Component directions must be inspected by bad type; overall clean-vs-bad AUC hides important failure modes.",
            "- This supports the revised scientific framing: SAS-Cert should be treated as a multi-dimensional certificate, not a fixed universal scalar score.",
            "",
            "## Decision",
            "",
            "`revise_scalar_score_before_training_expansion`",
            "",
            "Next action: define a backbone-aware or component-gated certificate rule from component directions, then test it in a small ST reliability setting before any new full expansion.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for backbone, meta in SOURCES.items():
        all_rows.extend(read_rows(backbone, Path(meta["score_dir"])))
    add_derived_scores(all_rows)
    components = BASE_COMPONENTS + ["score_artifact_gate_physio"]
    component_rows = component_auc(all_rows, components)
    bad_rows = bad_type_auc(all_rows, components)
    dist_rows = score_distributions(all_rows, components)
    conflict_rows = direction_conflicts([row for row in component_rows if row["comparison"] == "clean_vs_all_bad"])
    bad_conflict_rows = bad_type_direction_conflicts(bad_rows)
    best_rows = best_variants(all_rows)
    payload = {
        "status": "completed",
        "scope": {
            "dataset": "PhysioNetMI",
            "targets": TARGETS,
            "seeds": SEEDS,
            "sources": {k: {"score_dir": str(v["score_dir"]), "note": v["note"]} for k, v in SOURCES.items()},
            "uses_existing_score_rows_only": True,
            "no_retraining": True,
            "no_new_augmentation_generation": True,
            "target_test_labels_used": False,
        },
        "component_auc_by_backbone": component_rows,
        "component_auc_by_bad_type": bad_rows,
        "direction_conflict_table": conflict_rows,
        "bad_type_direction_conflict_table": bad_conflict_rows,
        "score_distribution_by_aug_type": dist_rows,
        "best_variants": best_rows,
        "decision": "revise_scalar_score_before_training_expansion",
    }
    write_csv(OUT_DIR / "component_auc_by_backbone.csv", component_rows)
    write_csv(OUT_DIR / "component_auc_by_bad_type.csv", bad_rows)
    write_csv(OUT_DIR / "direction_conflict_table.csv", conflict_rows)
    write_csv(OUT_DIR / "bad_type_direction_conflict_table.csv", bad_conflict_rows)
    write_csv(OUT_DIR / "score_distribution_by_aug_type.csv", dist_rows)
    write_csv(OUT_DIR / "best_variants.csv", best_rows)
    (OUT_DIR / "compact_cross_backbone_cert_direction_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "CROSS_BACKBONE_CERT_DIRECTION_AUDIT.md").write_text(make_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
