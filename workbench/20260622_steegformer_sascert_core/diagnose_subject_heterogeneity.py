#!/usr/bin/env python3
"""Subject-level heterogeneity diagnostics for the ST-SAS-Cert workbench."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_target_seed(path: Path) -> tuple[int, int]:
    match = re.search(r"target(\d+)_seed(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"cannot parse target/seed from {path}")
    return int(match.group(1)), int(match.group(2))


def pearson(x: list[float], y: list[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return None
    xa = xa[mask]
    ya = ya[mask]
    if float(xa.std()) == 0.0 or float(ya.std()) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def grouped_metrics(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, dict[str, str]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((int(row["target_subject"]), int(row["seed"])), {})[row["group"]] = row
    return grouped


def score_features(score_dir: Path) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for path in sorted(score_dir.glob("target*_seed*.csv")):
        target, seed = parse_target_seed(path)
        rows = read_csv(path)
        content = np.asarray([safe_float(r["content_score"]) for r in rows], dtype=np.float64)
        sas = np.asarray([safe_float(r["sas_score"]) for r in rows], dtype=np.float64)
        artifact = np.asarray([safe_float(r["artifact_risk_raw"]) for r in rows], dtype=np.float64)
        aug_types = [r["aug_type"] for r in rows]
        clean = np.asarray([t == "clean" for t in aug_types], dtype=bool)
        bad_artifact = np.asarray([t == "bad_artifact" for t in aug_types], dtype=bool)
        bad_content = np.asarray([t == "bad_content" for t in aug_types], dtype=bool)
        bad_physio = np.asarray([t == "bad_physio" for t in aug_types], dtype=bool)
        p90 = float(np.percentile(artifact, 90))
        reject_p90 = artifact >= p90
        out[(target, seed)] = {
            "mean_content_score": float(np.nanmean(content)),
            "std_content_score": float(np.nanstd(content)),
            "mean_sas_score": float(np.nanmean(sas)),
            "std_sas_score": float(np.nanstd(sas)),
            "mean_artifact_risk": float(np.nanmean(artifact)),
            "p90_artifact_risk": p90,
            "clean_mean_content": float(np.nanmean(content[clean])),
            "bad_artifact_mean_content": float(np.nanmean(content[bad_artifact])),
            "bad_content_mean_content": float(np.nanmean(content[bad_content])),
            "bad_physio_mean_content": float(np.nanmean(content[bad_physio])),
            "p90_clean_reject_rate": float((reject_p90 & clean).sum() / clean.sum()) if clean.sum() else float("nan"),
            "p90_bad_artifact_reject_rate": float((reject_p90 & bad_artifact).sum() / bad_artifact.sum()) if bad_artifact.sum() else float("nan"),
        }
    return out


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    metric_rows = read_csv(OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv")
    grouped = grouped_metrics(metric_rows)
    score_map = score_features(OUT_DIR / "score_rows" / args.score_tag)

    fold_rows: list[dict[str, object]] = []
    for (target, seed), group_map in sorted(grouped.items()):
        if not all(g in group_map for g in [args.primary_group, args.secondary_group, args.baseline]):
            continue
        base = group_map[args.baseline]
        primary = group_map[args.primary_group]
        secondary = group_map[args.secondary_group]
        row: dict[str, object] = {
            "target_subject": target,
            "seed": seed,
            "baseline_macro_f1": safe_float(base["macro_f1"]),
            "baseline_balanced_accuracy": safe_float(base["balanced_accuracy"]),
            "baseline_ece": safe_float(base["ece"]),
            "primary_delta_macro_f1": safe_float(primary["macro_f1"]) - safe_float(base["macro_f1"]),
            "primary_delta_balanced_accuracy": safe_float(primary["balanced_accuracy"]) - safe_float(base["balanced_accuracy"]),
            "primary_delta_ece": safe_float(primary["ece"]) - safe_float(base["ece"]),
            "primary_delta_nll": safe_float(primary["nll"]) - safe_float(base["nll"]),
            "secondary_delta_macro_f1": safe_float(secondary["macro_f1"]) - safe_float(base["macro_f1"]),
            "secondary_delta_balanced_accuracy": safe_float(secondary["balanced_accuracy"]) - safe_float(base["balanced_accuracy"]),
            "secondary_delta_ece": safe_float(secondary["ece"]) - safe_float(base["ece"]),
            "secondary_delta_nll": safe_float(secondary["nll"]) - safe_float(base["nll"]),
        }
        row.update(score_map.get((target, seed), {}))
        fold_rows.append(row)

    subject_rows: list[dict[str, object]] = []
    for target in sorted({int(r["target_subject"]) for r in fold_rows}):
        rows = [r for r in fold_rows if int(r["target_subject"]) == target]

        def avg(key: str) -> float:
            vals = [safe_float(r[key]) for r in rows]
            vals = [v for v in vals if math.isfinite(v)]
            return float(mean(vals)) if vals else float("nan")

        subject_rows.append(
            {
                "target_subject": target,
                "n_seeds": len(rows),
                "baseline_macro_f1": avg("baseline_macro_f1"),
                "baseline_balanced_accuracy": avg("baseline_balanced_accuracy"),
                "baseline_ece": avg("baseline_ece"),
                "primary_delta_macro_f1": avg("primary_delta_macro_f1"),
                "primary_delta_balanced_accuracy": avg("primary_delta_balanced_accuracy"),
                "primary_delta_ece": avg("primary_delta_ece"),
                "primary_delta_nll": avg("primary_delta_nll"),
                "primary_seed_win_rate_macro_f1": float(mean(safe_float(r["primary_delta_macro_f1"]) > 0 for r in rows)),
                "secondary_delta_macro_f1": avg("secondary_delta_macro_f1"),
                "secondary_delta_balanced_accuracy": avg("secondary_delta_balanced_accuracy"),
                "secondary_delta_ece": avg("secondary_delta_ece"),
                "secondary_delta_nll": avg("secondary_delta_nll"),
                "secondary_seed_win_rate_macro_f1": float(mean(safe_float(r["secondary_delta_macro_f1"]) > 0 for r in rows)),
                "mean_content_score": avg("mean_content_score"),
                "std_content_score": avg("std_content_score"),
                "mean_artifact_risk": avg("mean_artifact_risk"),
                "p90_artifact_risk": avg("p90_artifact_risk"),
                "p90_bad_artifact_reject_rate": avg("p90_bad_artifact_reject_rate"),
            }
        )

    primary_deltas = [safe_float(r["primary_delta_macro_f1"]) for r in subject_rows]
    secondary_deltas = [safe_float(r["secondary_delta_macro_f1"]) for r in subject_rows]
    summary = {
        "status": "completed",
        "output_tag": args.output_tag,
        "score_tag": args.score_tag,
        "primary_group": args.primary_group,
        "secondary_group": args.secondary_group,
        "baseline": args.baseline,
        "n_fold_rows": len(fold_rows),
        "n_subjects": len(subject_rows),
        "primary_subject_win_rate": float(mean(d > 0 for d in primary_deltas)),
        "secondary_subject_win_rate": float(mean(d > 0 for d in secondary_deltas)),
        "primary_mean_subject_delta_macro_f1": float(mean(primary_deltas)),
        "secondary_mean_subject_delta_macro_f1": float(mean(secondary_deltas)),
        "primary_losing_subjects": [int(r["target_subject"]) for r in subject_rows if safe_float(r["primary_delta_macro_f1"]) <= 0],
        "primary_winning_subjects": [int(r["target_subject"]) for r in subject_rows if safe_float(r["primary_delta_macro_f1"]) > 0],
        "secondary_losing_subjects": [int(r["target_subject"]) for r in subject_rows if safe_float(r["secondary_delta_macro_f1"]) <= 0],
        "secondary_winning_subjects": [int(r["target_subject"]) for r in subject_rows if safe_float(r["secondary_delta_macro_f1"]) > 0],
        "correlations_primary_delta": {
            "baseline_macro_f1": pearson(
                [safe_float(r["baseline_macro_f1"]) for r in subject_rows],
                [safe_float(r["primary_delta_macro_f1"]) for r in subject_rows],
            ),
            "baseline_ece": pearson(
                [safe_float(r["baseline_ece"]) for r in subject_rows],
                [safe_float(r["primary_delta_macro_f1"]) for r in subject_rows],
            ),
            "std_content_score": pearson(
                [safe_float(r["std_content_score"]) for r in subject_rows],
                [safe_float(r["primary_delta_macro_f1"]) for r in subject_rows],
            ),
            "mean_artifact_risk": pearson(
                [safe_float(r["mean_artifact_risk"]) for r in subject_rows],
                [safe_float(r["primary_delta_macro_f1"]) for r in subject_rows],
            ),
        },
    }

    if summary["primary_subject_win_rate"] < 0.65:
        decision = "subject_reliability_failed"
    else:
        decision = "subject_reliability_supported"
    summary["decision"] = decision
    return subject_rows, summary


def write_report(path: Path, subject_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    worst = sorted(subject_rows, key=lambda r: safe_float(r["primary_delta_macro_f1"]))[:5]
    best = sorted(subject_rows, key=lambda r: safe_float(r["primary_delta_macro_f1"]), reverse=True)[:5]
    lines = [
        "# Subject Heterogeneity Report",
        "",
        f"- Output tag: `{summary['output_tag']}`",
        f"- Primary: `{summary['primary_group']} vs {summary['baseline']}`",
        f"- Secondary: `{summary['secondary_group']} vs {summary['baseline']}`",
        f"- Subjects: `{summary['n_subjects']}`",
        "",
        "## Summary",
        "",
        f"- Primary subject win rate: `{summary['primary_subject_win_rate']:.4f}`",
        f"- Secondary subject win rate: `{summary['secondary_subject_win_rate']:.4f}`",
        f"- Primary mean subject delta Macro-F1: `{summary['primary_mean_subject_delta_macro_f1']:.6f}`",
        f"- Secondary mean subject delta Macro-F1: `{summary['secondary_mean_subject_delta_macro_f1']:.6f}`",
        f"- Decision: `{summary['decision']}`",
        "",
        "## Worst Primary Subjects",
        "",
        "| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            f"| {int(row['target_subject'])} | {safe_float(row['baseline_macro_f1']):.4f} | "
            f"{safe_float(row['primary_delta_macro_f1']):+.4f} | {safe_float(row['primary_delta_balanced_accuracy']):+.4f} | "
            f"{safe_float(row['primary_delta_ece']):+.4f} | {safe_float(row['std_content_score']):.4f} | {safe_float(row['mean_artifact_risk']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Primary Subjects",
            "",
            "| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best:
        lines.append(
            f"| {int(row['target_subject'])} | {safe_float(row['baseline_macro_f1']):.4f} | "
            f"{safe_float(row['primary_delta_macro_f1']):+.4f} | {safe_float(row['primary_delta_balanced_accuracy']):+.4f} | "
            f"{safe_float(row['primary_delta_ece']):+.4f} | {safe_float(row['std_content_score']):.4f} | {safe_float(row['mean_artifact_risk']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Correlations With Primary Delta Macro-F1",
            "",
        ]
    )
    for key, value in summary["correlations_primary_delta"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="st_source_tuned_full")
    parser.add_argument("--score-tag", default="st_source_tuned_full")
    parser.add_argument("--write-tag", default=None)
    parser.add_argument("--primary-group", default="SoftWeight_noReject_LS010")
    parser.add_argument("--secondary-group", default="SASCert_SoftAR_LS010")
    parser.add_argument("--baseline", default="NaiveAug_LS010")
    args = parser.parse_args()
    if args.write_tag is None:
        args.write_tag = args.output_tag
    return args


def main() -> None:
    args = parse_args()
    out = OUT_DIR / "subject_heterogeneity"
    out.mkdir(parents=True, exist_ok=True)
    subject_rows, summary = summarize(args)
    write_csv(out / f"subject_delta_table_{args.write_tag}.csv", subject_rows)
    (out / f"heterogeneity_summary_{args.write_tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / f"SUBJECT_HETEROGENEITY_REPORT_{args.write_tag}.md", subject_rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
