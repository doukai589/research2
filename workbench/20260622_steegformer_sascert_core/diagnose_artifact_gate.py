#!/usr/bin/env python3
"""Diagnose whether the artifact gate over-prunes ST source-tuned candidates."""

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


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def parse_target_seed(path: Path) -> tuple[int, int]:
    match = re.search(r"target(\d+)_seed(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"cannot parse target/seed from {path}")
    return int(match.group(1)), int(match.group(2))


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 2:
        return None
    xa = xa[mask]
    ya = ya[mask]
    if float(xa.std()) == 0.0 or float(ya.std()) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def load_pairs(path: Path) -> dict[tuple[int, int, str], dict[str, str]]:
    rows = read_csv(path)
    out: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["target_subject"]), int(row["seed"]), row["comparison"])
        out[key] = row
    return out


def summarize_gate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    score_dir = OUT_DIR / "score_rows" / args.score_tag
    score_files = sorted(score_dir.glob("target*_seed*.csv"))
    if not score_files:
        raise FileNotFoundError(f"no score rows under {score_dir}")

    pair_path = OUT_DIR / f"steegformer_physionetmi_paired_comparison_{args.output_tag}.csv"
    pairs = load_pairs(pair_path)

    fold_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    aggregate_by_type: dict[str, list[dict[str, object]]] = {}

    for path in score_files:
        target, seed = parse_target_seed(path)
        rows = read_csv(path)
        risks = np.asarray([safe_float(r["artifact_risk_raw"]) for r in rows], dtype=np.float64)
        contents = np.asarray([safe_float(r["content_score"]) for r in rows], dtype=np.float64)
        threshold = float(np.percentile(risks, args.artifact_reject_percentile))
        rejected = risks >= threshold
        aug_types = [r["aug_type"] for r in rows]
        is_clean = np.asarray([t == "clean" for t in aug_types], dtype=bool)
        is_bad_artifact = np.asarray([t == "bad_artifact" for t in aug_types], dtype=bool)
        is_bad_content = np.asarray([t == "bad_content" for t in aug_types], dtype=bool)
        is_bad_physio = np.asarray([t == "bad_physio" for t in aug_types], dtype=bool)

        def rate(mask: np.ndarray) -> float:
            denom = int(mask.sum())
            return float((rejected & mask).sum() / denom) if denom else float("nan")

        soft_cmp = pairs.get((target, seed, "SASCert_SoftAR_LS010-vs-SoftWeight_noReject_LS010"), {})
        naive_cmp = pairs.get((target, seed, "SASCert_SoftAR_LS010-vs-NaiveAug_LS010"), {})
        artifact_cmp = pairs.get((target, seed, "SASCert_SoftAR_LS010-vs-ArtifactReject_LS010"), {})

        fold_rows.append(
            {
                "target_subject": target,
                "seed": seed,
                "n_candidates": len(rows),
                "threshold_percentile": args.artifact_reject_percentile,
                "artifact_threshold": threshold,
                "reject_rate": float(rejected.mean()),
                "clean_reject_rate": rate(is_clean),
                "bad_artifact_reject_rate": rate(is_bad_artifact),
                "bad_content_reject_rate": rate(is_bad_content),
                "bad_physio_reject_rate": rate(is_bad_physio),
                "rejected_clean_fraction": float((rejected & is_clean).sum() / rejected.sum()) if rejected.sum() else 0.0,
                "rejected_bad_artifact_fraction": float((rejected & is_bad_artifact).sum() / rejected.sum()) if rejected.sum() else 0.0,
                "kept_bad_artifact_rate": float((~rejected & is_bad_artifact).sum() / is_bad_artifact.sum()) if is_bad_artifact.sum() else float("nan"),
                "mean_content_rejected": float(np.nanmean(contents[rejected])) if rejected.any() else float("nan"),
                "mean_content_kept": float(np.nanmean(contents[~rejected])) if (~rejected).any() else float("nan"),
                "delta_macro_f1_softar_vs_softweight": safe_float(soft_cmp.get("delta_macro_f1")),
                "delta_bacc_softar_vs_softweight": safe_float(soft_cmp.get("delta_balanced_accuracy")),
                "delta_ece_softar_vs_softweight": safe_float(soft_cmp.get("delta_ece")),
                "delta_macro_f1_softar_vs_naive": safe_float(naive_cmp.get("delta_macro_f1")),
                "delta_macro_f1_softar_vs_artifactreject": safe_float(artifact_cmp.get("delta_macro_f1")),
            }
        )

        for aug_type in sorted(set(aug_types)):
            mask = np.asarray([t == aug_type for t in aug_types], dtype=bool)
            row = {
                "target_subject": target,
                "seed": seed,
                "aug_type": aug_type,
                "n": int(mask.sum()),
                "reject_rate": rate(mask),
                "mean_artifact_risk": float(np.nanmean(risks[mask])),
                "mean_content_score": float(np.nanmean(contents[mask])),
            }
            type_rows.append(row)
            aggregate_by_type.setdefault(aug_type, []).append(row)

    def avg(key: str, rows: list[dict[str, object]] = fold_rows) -> float:
        vals = [safe_float(r[key]) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        return float(mean(vals)) if vals else float("nan")

    by_type_summary = {
        aug_type: {
            "mean_reject_rate": avg("reject_rate", rows),
            "mean_artifact_risk": avg("mean_artifact_risk", rows),
            "mean_content_score": avg("mean_content_score", rows),
        }
        for aug_type, rows in sorted(aggregate_by_type.items())
    }

    correlations = {
        "clean_reject_rate_vs_softar_minus_softweight_macro_f1": pearson(
            [safe_float(r["clean_reject_rate"]) for r in fold_rows],
            [safe_float(r["delta_macro_f1_softar_vs_softweight"]) for r in fold_rows],
        ),
        "bad_artifact_reject_rate_vs_softar_minus_softweight_macro_f1": pearson(
            [safe_float(r["bad_artifact_reject_rate"]) for r in fold_rows],
            [safe_float(r["delta_macro_f1_softar_vs_softweight"]) for r in fold_rows],
        ),
        "rejected_clean_fraction_vs_softar_minus_softweight_macro_f1": pearson(
            [safe_float(r["rejected_clean_fraction"]) for r in fold_rows],
            [safe_float(r["delta_macro_f1_softar_vs_softweight"]) for r in fold_rows],
        ),
    }

    summary = {
        "status": "completed",
        "score_tag": args.score_tag,
        "output_tag": args.output_tag,
        "write_tag": args.write_tag,
        "artifact_reject_percentile": args.artifact_reject_percentile,
        "n_folds": len(fold_rows),
        "mean_reject_rate": avg("reject_rate"),
        "mean_clean_reject_rate": avg("clean_reject_rate"),
        "mean_bad_artifact_reject_rate": avg("bad_artifact_reject_rate"),
        "mean_bad_content_reject_rate": avg("bad_content_reject_rate"),
        "mean_bad_physio_reject_rate": avg("bad_physio_reject_rate"),
        "mean_rejected_clean_fraction": avg("rejected_clean_fraction"),
        "mean_rejected_bad_artifact_fraction": avg("rejected_bad_artifact_fraction"),
        "mean_kept_bad_artifact_rate": avg("kept_bad_artifact_rate"),
        "mean_content_rejected_minus_kept": avg("mean_content_rejected") - avg("mean_content_kept"),
        "mean_softar_minus_softweight_macro_f1": avg("delta_macro_f1_softar_vs_softweight"),
        "mean_softar_minus_softweight_bacc": avg("delta_bacc_softar_vs_softweight"),
        "mean_softar_minus_softweight_ece": avg("delta_ece_softar_vs_softweight"),
        "mean_softar_minus_naive_macro_f1": avg("delta_macro_f1_softar_vs_naive"),
        "by_aug_type": by_type_summary,
        "correlations": correlations,
    }

    if summary["mean_clean_reject_rate"] > 0.05 and summary["mean_softar_minus_softweight_macro_f1"] <= 0:
        decision = "artifact_gate_overprunes_clean_or_useful_candidates"
    elif summary["mean_bad_artifact_reject_rate"] >= 0.8 and summary["mean_softar_minus_softweight_macro_f1"] > 0:
        decision = "artifact_gate_helpful"
    elif (
        summary["mean_rejected_clean_fraction"] <= 0.01
        and summary["mean_rejected_bad_artifact_fraction"] >= 0.95
        and summary["mean_bad_artifact_reject_rate"] < 0.8
    ):
        decision = "artifact_gate_precise_but_conservative"
    elif summary["mean_softar_minus_softweight_macro_f1"] <= 0 and summary["mean_softar_minus_softweight_ece"] < 0:
        decision = "artifact_gate_calibration_tradeoff"
    else:
        decision = "artifact_gate_inconclusive"
    summary["decision"] = decision
    return fold_rows, type_rows, summary


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Artifact Gate Diagnostic",
        "",
        f"- Score tag: `{summary['score_tag']}`",
        f"- Output tag: `{summary['output_tag']}`",
        f"- Folds: `{summary['n_folds']}`",
        f"- Artifact reject percentile: `{summary['artifact_reject_percentile']}`",
        "",
        "## Aggregate",
        "",
        f"- Mean reject rate: `{summary['mean_reject_rate']:.4f}`",
        f"- Mean clean reject rate: `{summary['mean_clean_reject_rate']:.4f}`",
        f"- Mean BadArtifact reject rate: `{summary['mean_bad_artifact_reject_rate']:.4f}`",
        f"- Mean rejected clean fraction: `{summary['mean_rejected_clean_fraction']:.4f}`",
        f"- Mean rejected BadArtifact fraction: `{summary['mean_rejected_bad_artifact_fraction']:.4f}`",
        f"- Mean kept BadArtifact rate: `{summary['mean_kept_bad_artifact_rate']:.4f}`",
        f"- Mean SoftAR - SoftWeight Macro-F1: `{summary['mean_softar_minus_softweight_macro_f1']:.6f}`",
        f"- Mean SoftAR - SoftWeight ECE: `{summary['mean_softar_minus_softweight_ece']:.6f}`",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "## By Augmentation Type",
        "",
        "| Aug Type | Reject Rate | Artifact Risk | Content Score |",
        "|---|---:|---:|---:|",
    ]
    for aug_type, stats in summary["by_aug_type"].items():
        lines.append(
            f"| {aug_type} | {stats['mean_reject_rate']:.4f} | {stats['mean_artifact_risk']:.4f} | {stats['mean_content_score']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-tag", default="st_source_tuned_full")
    parser.add_argument("--output-tag", default="st_source_tuned_full")
    parser.add_argument("--write-tag", default=None)
    parser.add_argument("--artifact-reject-percentile", type=float, default=90.0)
    args = parser.parse_args()
    if args.write_tag is None:
        suffix = str(args.artifact_reject_percentile).replace(".", "p")
        args.write_tag = f"{args.output_tag}_p{suffix}"
    return args


def main() -> None:
    args = parse_args()
    out = OUT_DIR / "artifact_gate_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    fold_rows, type_rows, summary = summarize_gate(args)
    tag = args.write_tag
    write_csv(out / f"artifact_gate_fold_stats_{tag}.csv", fold_rows)
    write_csv(out / f"artifact_gate_by_aug_type_{tag}.csv", type_rows)
    (out / f"artifact_gate_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / f"ARTIFACT_GATE_DIAGNOSTIC_{tag}.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
