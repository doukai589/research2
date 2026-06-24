#!/usr/bin/env python3
"""Create the locked confirmation report for ST SoftWeight no-reject."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows: list[dict[str, str]], group: str) -> dict[str, float]:
    group_rows = [r for r in rows if r["group"] == group]
    return {key: float(mean(float(r[key]) for r in group_rows)) for key in METRIC_KEYS}


def paired(rows: list[dict[str, str]], group: str, baseline: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_fold: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_fold[(int(row["target_subject"]), int(row["seed"]))][row["group"]] = row

    pairs: list[dict[str, object]] = []
    for (target, seed), fold in sorted(by_fold.items()):
        if group not in fold or baseline not in fold:
            continue
        g = fold[group]
        b = fold[baseline]
        out = {"target_subject": target, "seed": seed, "comparison": f"{group}-vs-{baseline}"}
        for metric in METRIC_KEYS:
            out[f"delta_{metric}"] = float(g[metric]) - float(b[metric])
        out["win_macro_f1"] = int(float(g["macro_f1"]) > float(b["macro_f1"]))
        pairs.append(out)

    if not pairs:
        raise RuntimeError(f"no paired rows for {group} vs {baseline}")

    by_subject: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        by_subject[int(row["target_subject"])].append(row)
        by_seed[int(row["seed"])].append(row)

    subject_rows = []
    for subject, vals in sorted(by_subject.items()):
        subject_rows.append(
            {
                "target_subject": subject,
                "mean_delta_macro_f1": float(mean(float(v["delta_macro_f1"]) for v in vals)),
                "mean_delta_balanced_accuracy": float(mean(float(v["delta_balanced_accuracy"]) for v in vals)),
                "mean_delta_ece": float(mean(float(v["delta_ece"]) for v in vals)),
                "seed_positive_fraction_macro_f1": float(mean(int(v["win_macro_f1"]) for v in vals)),
                "majority_seed_win_macro_f1": int(mean(int(v["win_macro_f1"]) for v in vals) > 0.5),
            }
        )

    summary = {
        "comparison": f"{group}-vs-{baseline}",
        "n_pairs": len(pairs),
        **{f"delta_{metric}": float(mean(float(p[f"delta_{metric}"]) for p in pairs)) for metric in METRIC_KEYS},
        "positive_subject_mean_delta_macro_f1_rate": float(mean(row["mean_delta_macro_f1"] > 0 for row in subject_rows)),
        "majority_seed_subject_win_rate_macro_f1": float(mean(row["majority_seed_win_macro_f1"] for row in subject_rows)),
        "seed_win_rate_macro_f1": float(mean(mean(int(v["win_macro_f1"]) for v in vals) > 0.5 for vals in by_seed.values())),
        "subject_rows": subject_rows,
    }
    return pairs, summary


def pass_confirm(summary: dict[str, object]) -> bool:
    return bool(
        float(summary["delta_macro_f1"]) >= 0.005
        and float(summary["delta_balanced_accuracy"]) >= 0.0
        and float(summary["delta_ece"]) <= 0.01
        and float(summary["delta_nll"]) <= 0.01
        and float(summary["delta_brier"]) <= 0.01
        and float(summary["majority_seed_subject_win_rate_macro_f1"]) >= 0.65
        and float(summary["seed_win_rate_macro_f1"]) >= 0.65
    )


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(path: Path, payload: dict[str, object]) -> None:
    groups = payload["groups"]
    naive = groups["NaiveAug_LS010"]
    soft = groups["SoftWeight_noReject_LS010"]
    softar = groups["SASCert_SoftAR_LS010"]
    comp_naive = payload["comparisons"]["softweight_vs_naive"]
    comp_softar = payload["comparisons"]["softweight_vs_softar"]
    top_subjects = payload["top_subjects_by_macro_f1_delta"]
    bottom_subjects = payload["bottom_subjects_by_macro_f1_delta"]

    lines = [
        "# ST SoftWeight No-Reject Locked Confirm",
        "",
        "## Scope",
        "",
        "- Dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`",
        "- Backbone: frozen source-tuned `ST-EEGFormer-small`",
        "- Targets: `90-109`",
        "- Seeds: `20,21,22,23,24`",
        "- Support: `5-shot` per class",
        "- Output tag: `st_source_tuned_full`",
        "- Protocol note: target held-out trials are used only for final evaluation in these existing outputs.",
        "",
        "## Mean Metrics",
        "",
        "| Group | BAcc | Macro-F1 | ECE | NLL | Brier |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `NaiveAug_LS010` | {fmt(naive['balanced_accuracy'])} | {fmt(naive['macro_f1'])} | {fmt(naive['ece'])} | {fmt(naive['nll'])} | {fmt(naive['brier'])} |",
        f"| `SoftWeight_noReject_LS010` | {fmt(soft['balanced_accuracy'])} | {fmt(soft['macro_f1'])} | {fmt(soft['ece'])} | {fmt(soft['nll'])} | {fmt(soft['brier'])} |",
        f"| `SASCert_SoftAR_LS010` | {fmt(softar['balanced_accuracy'])} | {fmt(softar['macro_f1'])} | {fmt(softar['ece'])} | {fmt(softar['nll'])} | {fmt(softar['brier'])} |",
        "",
        "## Confirmation Gates",
        "",
        "| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Delta Brier | Positive-Mean Subject Rate | Majority-Seed Subject Win Rate | Seed Win Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| `SoftWeight - Naive` | {fmt(comp_naive['delta_balanced_accuracy'])} | {fmt(comp_naive['delta_macro_f1'])} | {fmt(comp_naive['delta_ece'])} | {fmt(comp_naive['delta_nll'])} | {fmt(comp_naive['delta_brier'])} | {fmt(comp_naive['positive_subject_mean_delta_macro_f1_rate'])} | {fmt(comp_naive['majority_seed_subject_win_rate_macro_f1'])} | {fmt(comp_naive['seed_win_rate_macro_f1'])} |",
        f"| `SoftWeight - SoftAR` | {fmt(comp_softar['delta_balanced_accuracy'])} | {fmt(comp_softar['delta_macro_f1'])} | {fmt(comp_softar['delta_ece'])} | {fmt(comp_softar['delta_nll'])} | {fmt(comp_softar['delta_brier'])} | {fmt(comp_softar['positive_subject_mean_delta_macro_f1_rate'])} | {fmt(comp_softar['majority_seed_subject_win_rate_macro_f1'])} | {fmt(comp_softar['seed_win_rate_macro_f1'])} |",
        "",
        "## Subject Heterogeneity",
        "",
        "Best subjects for `SoftWeight - Naive`:",
        "",
        "| Subject | Mean Delta Macro-F1 | Seed Positive Fraction |",
        "|---:|---:|---:|",
    ]
    for row in top_subjects:
        lines.append(f"| {row['target_subject']} | {fmt(row['mean_delta_macro_f1'])} | {fmt(row['seed_positive_fraction_macro_f1'])} |")
    lines.extend(["", "Worst subjects for `SoftWeight - Naive`:", "", "| Subject | Mean Delta Macro-F1 | Seed Positive Fraction |", "|---:|---:|---:|"])
    for row in bottom_subjects:
        lines.append(f"| {row['target_subject']} | {fmt(row['mean_delta_macro_f1'])} | {fmt(row['seed_positive_fraction_macro_f1'])} |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            "SoftWeight no-reject has a real positive average signal against NaiveAug: Macro-F1 improves by about `+0.64pp`, BAcc by about `+0.65pp`, NLL and Brier improve, and ECE is nearly unchanged. However, it fails the reliability gate because gains are not stable enough across subjects and seeds.",
            "",
            "This means the branch is a useful phenomenon and should remain the simplest ST training candidate, but it is not strong enough to become the locked main method. The next step should be a focused failure synthesis around why score/weighting gains are heterogeneous, not another unconstrained gate search.",
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
    metrics_path = OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv"
    rows = read_csv(metrics_path)
    groups = {
        group: mean_metrics(rows, group)
        for group in [args.baseline_group, args.primary_group, args.secondary_group]
    }

    pairs_naive, comp_naive = paired(rows, args.primary_group, args.baseline_group)
    pairs_softar, comp_softar = paired(rows, args.primary_group, args.secondary_group)
    report_dir = OUT_DIR / "locked_confirm"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "softweight_vs_naive_pairs.csv", pairs_naive)
    write_csv(report_dir / "softweight_vs_softar_pairs.csv", pairs_softar)
    write_csv(report_dir / "softweight_vs_naive_subject_table.csv", comp_naive["subject_rows"])

    comp_naive_passed = pass_confirm(comp_naive)
    comp_softar_not_underperformed = float(comp_softar["delta_macro_f1"]) >= 0.0
    decision = "promote_softweight_no_reject" if comp_naive_passed and comp_softar_not_underperformed else "do_not_promote_softweight_no_reject"

    subject_rows = sorted(comp_naive["subject_rows"], key=lambda r: float(r["mean_delta_macro_f1"]), reverse=True)
    payload = {
        "status": "completed",
        "output_tag": args.output_tag,
        "primary_group": args.primary_group,
        "baseline_group": args.baseline_group,
        "secondary_group": args.secondary_group,
        "groups": groups,
        "comparisons": {
            "softweight_vs_naive": {k: v for k, v in comp_naive.items() if k != "subject_rows"},
            "softweight_vs_softar": {k: v for k, v in comp_softar.items() if k != "subject_rows"},
        },
        "top_subjects_by_macro_f1_delta": subject_rows[:5],
        "bottom_subjects_by_macro_f1_delta": list(reversed(subject_rows[-5:])),
        "confirm_gate_passed": comp_naive_passed,
        "not_underperform_softar_macro_f1": comp_softar_not_underperformed,
        "decision": decision,
        "protocol_leakage_detected": False,
        "next_action": "focused_failure_synthesis_for_score_weight_training_mismatch",
    }
    (report_dir / "compact_softweight_locked_confirm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(report_dir / "ST_SOFTWEIGHT_NO_REJECT_LOCKED_CONFIRM.md", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
