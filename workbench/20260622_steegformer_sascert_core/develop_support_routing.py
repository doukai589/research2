#!/usr/bin/env python3
"""Develop support-only routing rules on validation subjects.

This script uses validation-subject evaluation labels only to develop the rule.
It must not be run on final target subjects to tune thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
GROUPS = ["NaiveAug_LS010", "SoftWeight_noReject_LS010", "SASCert_SoftAR_LS010"]
METRICS = ["balanced_accuracy", "macro_f1", "ece", "nll", "brier"]


@dataclass(frozen=True)
class Rule:
    kind: str
    group: str | None = None
    feature: str | None = None
    threshold: float | None = None
    le_group: str | None = None
    gt_group: str | None = None

    def name(self) -> str:
        if self.kind == "always":
            return f"always:{self.group}"
        return f"threshold:{self.feature}<={self.threshold:.6g}?{self.le_group}:{self.gt_group}"


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


def load_metrics(output_tag: str) -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    rows = read_csv(OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{output_tag}.csv")
    out: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
    for row in rows:
        key = (int(row["target_subject"]), int(row["seed"]))
        out.setdefault(key, {})[row["group"]] = {metric: safe_float(row[metric]) for metric in METRICS}
    return out


def load_features(score_tag: str) -> dict[tuple[int, int], dict[str, float]]:
    score_dir = OUT_DIR / "score_rows" / score_tag
    out: dict[tuple[int, int], dict[str, float]] = {}
    for path in sorted(score_dir.glob("target*_seed*.csv")):
        target, seed = parse_target_seed(path)
        rows = read_csv(path)
        content = np.asarray([safe_float(r["content_score"]) for r in rows], dtype=np.float64)
        style = np.asarray([safe_float(r["style_score"]) for r in rows], dtype=np.float64)
        physio = np.asarray([safe_float(r["physio_score"]) for r in rows], dtype=np.float64)
        sas = np.asarray([safe_float(r["sas_score"]) for r in rows], dtype=np.float64)
        artifact = np.asarray([safe_float(r["artifact_risk_raw"]) for r in rows], dtype=np.float64)
        out[(target, seed)] = {
            "mean_content": float(np.nanmean(content)),
            "std_content": float(np.nanstd(content)),
            "mean_style": float(np.nanmean(style)),
            "std_style": float(np.nanstd(style)),
            "mean_physio": float(np.nanmean(physio)),
            "std_physio": float(np.nanstd(physio)),
            "mean_sas": float(np.nanmean(sas)),
            "std_sas": float(np.nanstd(sas)),
            "mean_artifact_risk": float(np.nanmean(artifact)),
            "std_artifact_risk": float(np.nanstd(artifact)),
            "p80_artifact_risk": float(np.nanpercentile(artifact, 80)),
            "p90_artifact_risk": float(np.nanpercentile(artifact, 90)),
            "max_artifact_risk": float(np.nanmax(artifact)),
        }
    return out


def joined_rows(output_tag: str, score_tag: str) -> list[dict[str, object]]:
    metrics = load_metrics(output_tag)
    features = load_features(score_tag)
    rows: list[dict[str, object]] = []
    for key, group_metrics in sorted(metrics.items()):
        if not all(group in group_metrics for group in GROUPS) or key not in features:
            continue
        target, seed = key
        row: dict[str, object] = {"target_subject": target, "seed": seed}
        row.update(features[key])
        for group in GROUPS:
            for metric, value in group_metrics[group].items():
                row[f"{group}__{metric}"] = value
        best = max(GROUPS, key=lambda group: group_metrics[group]["macro_f1"])
        row["best_group_macro_f1"] = best
        rows.append(row)
    return rows


def choose(rule: Rule, row: dict[str, object]) -> str:
    if rule.kind == "always":
        assert rule.group is not None
        return rule.group
    assert rule.feature is not None and rule.threshold is not None and rule.le_group and rule.gt_group
    value = safe_float(row[rule.feature])
    return rule.le_group if value <= rule.threshold else rule.gt_group


def eval_rule(rule: Rule, rows: list[dict[str, object]]) -> dict[str, float]:
    selected = [choose(rule, row) for row in rows]
    result = {
        "macro_f1": float(mean(safe_float(row[f"{group}__macro_f1"]) for row, group in zip(rows, selected))),
        "balanced_accuracy": float(mean(safe_float(row[f"{group}__balanced_accuracy"]) for row, group in zip(rows, selected))),
        "ece": float(mean(safe_float(row[f"{group}__ece"]) for row, group in zip(rows, selected))),
        "nll": float(mean(safe_float(row[f"{group}__nll"]) for row, group in zip(rows, selected))),
        "brier": float(mean(safe_float(row[f"{group}__brier"]) for row, group in zip(rows, selected))),
    }
    result["selection_NaiveAug_LS010"] = float(selected.count("NaiveAug_LS010") / len(selected))
    result["selection_SoftWeight_noReject_LS010"] = float(selected.count("SoftWeight_noReject_LS010") / len(selected))
    result["selection_SASCert_SoftAR_LS010"] = float(selected.count("SASCert_SoftAR_LS010") / len(selected))
    return result


def candidate_rules(rows: list[dict[str, object]]) -> list[Rule]:
    rules = [Rule(kind="always", group=group) for group in GROUPS]
    feature_names = [
        "mean_content",
        "std_content",
        "mean_style",
        "std_style",
        "mean_physio",
        "std_physio",
        "mean_sas",
        "std_sas",
        "mean_artifact_risk",
        "std_artifact_risk",
        "p80_artifact_risk",
        "p90_artifact_risk",
        "max_artifact_risk",
    ]
    for feature in feature_names:
        values = sorted({safe_float(row[feature]) for row in rows if math.isfinite(safe_float(row[feature]))})
        if len(values) < 2:
            continue
        if len(values) > 20:
            qs = np.linspace(0.1, 0.9, 17)
            thresholds = sorted(set(float(np.quantile(values, q)) for q in qs))
        else:
            thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
        for threshold in thresholds:
            for le_group in GROUPS:
                for gt_group in GROUPS:
                    if le_group == gt_group:
                        continue
                    rules.append(Rule(kind="threshold", feature=feature, threshold=threshold, le_group=le_group, gt_group=gt_group))
    return rules


def best_rule(rows: list[dict[str, object]]) -> tuple[Rule, dict[str, float]]:
    best: tuple[Rule, dict[str, float]] | None = None
    for rule in candidate_rules(rows):
        score = eval_rule(rule, rows)
        if best is None:
            best = (rule, score)
            continue
        _, best_score = best
        if (score["macro_f1"], -score["ece"], -score["nll"]) > (
            best_score["macro_f1"],
            -best_score["ece"],
            -best_score["nll"],
        ):
            best = (rule, score)
    assert best is not None
    return best


def loso_cv(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    cv_rows: list[dict[str, object]] = []
    subjects = sorted({int(row["target_subject"]) for row in rows})
    for subject in subjects:
        train = [row for row in rows if int(row["target_subject"]) != subject]
        test = [row for row in rows if int(row["target_subject"]) == subject]
        rule, train_score = best_rule(train)
        test_score = eval_rule(rule, test)
        cv_rows.append(
            {
                "heldout_subject": subject,
                "rule": rule.name(),
                "train_macro_f1": train_score["macro_f1"],
                "test_macro_f1": test_score["macro_f1"],
                "test_balanced_accuracy": test_score["balanced_accuracy"],
                "test_ece": test_score["ece"],
                "test_nll": test_score["nll"],
                "test_brier": test_score["brier"],
                "selection_NaiveAug_LS010": test_score["selection_NaiveAug_LS010"],
                "selection_SoftWeight_noReject_LS010": test_score["selection_SoftWeight_noReject_LS010"],
                "selection_SASCert_SoftAR_LS010": test_score["selection_SASCert_SoftAR_LS010"],
            }
        )
    summary = {
        "macro_f1": float(mean(row["test_macro_f1"] for row in cv_rows)),
        "balanced_accuracy": float(mean(row["test_balanced_accuracy"] for row in cv_rows)),
        "ece": float(mean(row["test_ece"] for row in cv_rows)),
        "nll": float(mean(row["test_nll"] for row in cv_rows)),
        "brier": float(mean(row["test_brier"] for row in cv_rows)),
    }
    return cv_rows, summary


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Support Routing Dev Report",
        "",
        f"- Output tag: `{summary['output_tag']}`",
        f"- Score tag: `{summary['score_tag']}`",
        f"- Dev folds: `{summary['n_rows']}`",
        f"- Dev subjects: `{summary['n_subjects']}`",
        "",
        "## Baselines",
        "",
        "| Strategy | Macro-F1 | BAcc | ECE | NLL | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["constant_strategies"].items():
        lines.append(
            f"| {name} | {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['ece']:.4f} | {metrics['nll']:.4f} | {metrics['brier']:.4f} |"
        )
    cv = summary["loso_cv_routing"]
    lines.extend(
        [
            f"| LOSO support routing | {cv['macro_f1']:.4f} | {cv['balanced_accuracy']:.4f} | {cv['ece']:.4f} | {cv['nll']:.4f} | {cv['brier']:.4f} |",
            "",
            "## Full-Dev Best Rule",
            "",
            f"- Rule: `{summary['full_dev_best_rule']['rule']}`",
            f"- Macro-F1: `{summary['full_dev_best_rule']['macro_f1']:.6f}`",
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            summary["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="st_source_tuned_routing_dev")
    parser.add_argument("--score-tag", default="st_source_tuned_routing_dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = OUT_DIR / "support_routing_dev"
    out.mkdir(parents=True, exist_ok=True)
    rows = joined_rows(args.output_tag, args.score_tag)
    write_csv(out / "routing_features.csv", rows)

    constants = {group: eval_rule(Rule(kind="always", group=group), rows) for group in GROUPS}
    cv_rows, cv_summary = loso_cv(rows)
    write_csv(out / "routing_loso_subject_cv.csv", cv_rows)
    full_rule, full_score = best_rule(rows)
    best_constant_name, best_constant = max(constants.items(), key=lambda item: item[1]["macro_f1"])

    improves_over_best_constant = cv_summary["macro_f1"] > best_constant["macro_f1"]
    ece_ok = cv_summary["ece"] <= constants["NaiveAug_LS010"]["ece"] + 0.01
    if improves_over_best_constant and ece_ok:
        decision = "freeze_candidate_rule_for_final_test"
        interpretation = "LOSO support routing beats the best constant dev strategy and passes the ECE guard."
    else:
        decision = "do_not_freeze_routing_rule"
        interpretation = (
            "LOSO support routing does not beat the best constant dev strategy under the current "
            "support-only features. Keep this as diagnostic evidence; do not apply a routed method to "
            "final target subjects yet."
        )

    summary = {
        "status": "completed",
        "output_tag": args.output_tag,
        "score_tag": args.score_tag,
        "n_rows": len(rows),
        "n_subjects": len({int(row["target_subject"]) for row in rows}),
        "constant_strategies": constants,
        "best_constant_strategy": best_constant_name,
        "loso_cv_routing": cv_summary,
        "full_dev_best_rule": {"rule": full_rule.name(), **full_score},
        "decision": decision,
        "interpretation": interpretation,
    }
    (out / "routing_rule.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / "SUPPORT_ROUTING_DEV_REPORT.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
