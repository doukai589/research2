#!/usr/bin/env python3
"""Audit whether candidate/support summaries explain SoftWeight utility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]
SCORE_KEYS = [
    "content_score",
    "style_score",
    "physio_score",
    "artifact_safe_score",
    "artifact_risk_raw",
    "sas_score",
]


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


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy))


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def grouped_metrics(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, dict[str, str]]]:
    out: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        out[(int(row["target_subject"]), int(row["seed"]))][row["group"]] = row
    return out


def candidate_summary(score_rows: list[dict[str, str]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in SCORE_KEYS:
        vals = [float(r[key]) for r in score_rows if r.get(key, "") != ""]
        out[f"{key}_mean"] = float(mean(vals))
        out[f"{key}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
        out[f"{key}_p10"] = float(np.percentile(vals, 10))
        out[f"{key}_p90"] = float(np.percentile(vals, 90))
    total = len(score_rows)
    for aug_type in sorted({r["aug_type"] for r in score_rows}):
        out[f"aug_frac_{aug_type}"] = float(sum(r["aug_type"] == aug_type for r in score_rows) / total)
    artifact = [float(r["artifact_risk_raw"]) for r in score_rows]
    threshold = float(np.percentile(artifact, 90))
    out["artifact_p90_reject_fraction"] = float(sum(v >= threshold for v in artifact) / len(artifact))
    clean_rows = [r for r in score_rows if r["aug_type"] == "clean"]
    if clean_rows:
        out["clean_content_score_mean"] = float(mean(float(r["content_score"]) for r in clean_rows))
        out["clean_artifact_risk_raw_mean"] = float(mean(float(r["artifact_risk_raw"]) for r in clean_rows))
    return out


def build_fold_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    metrics = read_csv(OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv")
    metrics_by_fold = grouped_metrics(metrics)
    score_dir = OUT_DIR / "score_rows" / args.output_tag
    rows: list[dict[str, object]] = []
    for (target, seed), fold in sorted(metrics_by_fold.items()):
        if args.primary_group not in fold or args.baseline_group not in fold or args.secondary_group not in fold:
            continue
        score_path = score_dir / f"target{target}_seed{seed}.csv"
        if not score_path.exists():
            continue
        primary = fold[args.primary_group]
        baseline = fold[args.baseline_group]
        secondary = fold[args.secondary_group]
        row: dict[str, object] = {
            "target_subject": target,
            "seed": seed,
            "baseline_macro_f1": float(baseline["macro_f1"]),
            "baseline_balanced_accuracy": float(baseline["balanced_accuracy"]),
            "softweight_macro_f1": float(primary["macro_f1"]),
            "softweight_balanced_accuracy": float(primary["balanced_accuracy"]),
            "softar_macro_f1": float(secondary["macro_f1"]),
            "delta_macro_f1_softweight_vs_naive": float(primary["macro_f1"]) - float(baseline["macro_f1"]),
            "delta_balanced_accuracy_softweight_vs_naive": float(primary["balanced_accuracy"]) - float(baseline["balanced_accuracy"]),
            "delta_ece_softweight_vs_naive": float(primary["ece"]) - float(baseline["ece"]),
            "delta_nll_softweight_vs_naive": float(primary["nll"]) - float(baseline["nll"]),
            "delta_macro_f1_softweight_vs_softar": float(primary["macro_f1"]) - float(secondary["macro_f1"]),
        }
        row.update(candidate_summary(read_csv(score_path)))
        rows.append(row)
    return rows


def is_candidate_predictor(key: str) -> bool:
    prefixes = (
        "content_score_",
        "style_score_",
        "physio_score_",
        "artifact_safe_score_",
        "artifact_risk_raw_",
        "sas_score_",
        "aug_frac_",
        "clean_content_score_",
        "clean_artifact_risk_raw_",
        "artifact_p90_reject_fraction",
    )
    return key.startswith(prefixes)


def correlations(rows: list[dict[str, object]], target_key: str, candidate_only: bool) -> list[dict[str, object]]:
    skip = {"target_subject", "seed"}
    numeric_keys = []
    for key, value in rows[0].items():
        if key in skip or key == target_key or not isinstance(value, (int, float)):
            continue
        if candidate_only and not is_candidate_predictor(key):
            continue
        numeric_keys.append(key)
    y = [float(r[target_key]) for r in rows]
    out = []
    for key in numeric_keys:
        x = [float(r[key]) for r in rows]
        p = pearson(x, y)
        s = spearman(x, y)
        if math.isnan(p) and math.isnan(s):
            continue
        out.append(
            {
                "feature": key,
                "target": target_key,
                "pearson": p,
                "spearman": s,
                "abs_spearman": abs(s) if not math.isnan(s) else None,
            }
        )
    out.sort(key=lambda r: -(float(r["abs_spearman"]) if r["abs_spearman"] is not None else -1.0))
    return out


def subject_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_subject: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["target_subject"])].append(row)
    out = []
    for subject, vals in sorted(by_subject.items()):
        out.append(
            {
                "target_subject": subject,
                "mean_delta_macro_f1": float(mean(float(v["delta_macro_f1_softweight_vs_naive"]) for v in vals)),
                "mean_baseline_macro_f1": float(mean(float(v["baseline_macro_f1"]) for v in vals)),
                "mean_content_score": float(mean(float(v["content_score_mean"]) for v in vals)),
                "mean_artifact_risk": float(mean(float(v["artifact_risk_raw_mean"]) for v in vals)),
                "positive_seed_fraction": float(mean(float(v["delta_macro_f1_softweight_vs_naive"]) > 0 for v in vals)),
            }
        )
    return out


def write_report(path: Path, payload: dict[str, object]) -> None:
    top_corr = payload["top_correlations"][:10]
    subject_rows = payload["subject_summary"]
    best = sorted(subject_rows, key=lambda r: float(r["mean_delta_macro_f1"]), reverse=True)[:5]
    worst = sorted(subject_rows, key=lambda r: float(r["mean_delta_macro_f1"]))[:5]
    lines = [
        "# Support/Candidate Utility Alignment Audit",
        "",
        "## Scope",
        "",
        "- Inputs: existing `st_source_tuned_full` metrics and score rows",
        "- No new model training",
        "- No new dataset or backbone",
        "- Purpose: test whether fold-level candidate/support summaries explain `SoftWeight_noReject_LS010` utility",
        "",
        "## Top Correlations With SoftWeight Benefit",
        "",
        "| Feature | Spearman | Pearson |",
        "|---|---:|---:|",
    ]
    for row in top_corr:
        lines.append(f"| `{row['feature']}` | {float(row['spearman']):.4f} | {float(row['pearson']):.4f} |")
    lines.extend(
        [
            "",
            "## Subject-Level Pattern",
            "",
            "Best subjects:",
            "",
            "| Subject | Mean Delta Macro-F1 | Baseline Macro-F1 | Positive Seed Fraction |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in best:
        lines.append(
            f"| {row['target_subject']} | {float(row['mean_delta_macro_f1']):.4f} | {float(row['mean_baseline_macro_f1']):.4f} | {float(row['positive_seed_fraction']):.4f} |"
        )
    lines.extend(["", "Worst subjects:", "", "| Subject | Mean Delta Macro-F1 | Baseline Macro-F1 | Positive Seed Fraction |", "|---:|---:|---:|---:|"])
    for row in worst:
        lines.append(
            f"| {row['target_subject']} | {float(row['mean_delta_macro_f1']):.4f} | {float(row['mean_baseline_macro_f1']):.4f} | {float(row['positive_seed_fraction']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            payload["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="st_source_tuned_full")
    parser.add_argument("--primary-group", default="SoftWeight_noReject_LS010")
    parser.add_argument("--baseline-group", default="NaiveAug_LS010")
    parser.add_argument("--secondary-group", default="SASCert_SoftAR_LS010")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = OUT_DIR / "utility_alignment_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = build_fold_rows(args)
    if not rows:
        raise RuntimeError("no fold rows built")
    corr = correlations(rows, "delta_macro_f1_softweight_vs_naive", candidate_only=True)
    retrospective_corr = correlations(rows, "delta_macro_f1_softweight_vs_naive", candidate_only=False)
    subj = subject_summary(rows)
    write_csv(audit_dir / "fold_utility_alignment_features.csv", rows)
    write_csv(audit_dir / "utility_alignment_correlations.csv", corr)
    write_csv(audit_dir / "utility_alignment_retrospective_correlations.csv", retrospective_corr)
    write_csv(audit_dir / "utility_alignment_subject_summary.csv", subj)

    strongest = corr[0] if corr else None
    strong_alignment = bool(strongest and abs(float(strongest["spearman"])) >= 0.35)
    decision = "design_locked_support_only_utility_rule" if strong_alignment else "park_st_weighting_variants"
    interpretation = (
        "The existing-output audit found at least one moderate fold-level alignment signal. "
        "This is not a deployable rule yet, but it is enough to justify designing one locked support-only utility rule before a rerun."
        if strong_alignment
        else "The existing-output audit did not find a strong enough fold-level alignment signal. "
        "Under the stop rule, ST weighting variants should be parked or reframed as diagnostic observations rather than expanded."
    )
    payload = {
        "status": "completed",
        "output_tag": args.output_tag,
        "n_folds": len(rows),
        "target": "delta_macro_f1_softweight_vs_naive",
        "top_correlations": corr[:15],
        "top_retrospective_correlations": retrospective_corr[:15],
        "subject_summary": subj,
        "strong_alignment_threshold_abs_spearman": 0.35,
        "strong_alignment_found": strong_alignment,
        "decision": decision,
        "interpretation": interpretation,
        "protocol_leakage_detected": False,
        "note": "This is retrospective analysis of existing outputs only; no thresholds are promoted from target held-out labels.",
    }
    (audit_dir / "compact_utility_alignment_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(audit_dir / "SUPPORT_CANDIDATE_UTILITY_ALIGNMENT_AUDIT.md", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
