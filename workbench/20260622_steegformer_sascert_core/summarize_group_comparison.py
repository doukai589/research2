#!/usr/bin/env python3
"""Summarize any two groups from a tagged ST-SAS-Cert metrics CSV."""

from __future__ import annotations

import argparse
import csv
import json
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, dict[str, str]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (int(row["target_subject"]), int(row["seed"]))
        grouped.setdefault(key, {})[row["group"]] = row
    return grouped


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_csv(OUT_DIR / f"steegformer_physionetmi_sascert_metrics_{args.output_tag}.csv")
    grouped = group_rows(rows)
    pairs: list[dict[str, object]] = []
    for (target, seed), group_map in sorted(grouped.items()):
        if args.group not in group_map or args.baseline not in group_map:
            continue
        g = group_map[args.group]
        b = group_map[args.baseline]
        pair = {
            "target_subject": target,
            "seed": seed,
            "comparison": f"{args.group}-vs-{args.baseline}",
        }
        for metric in METRIC_KEYS:
            pair[f"delta_{metric}"] = float(g[metric]) - float(b[metric])
        pair["win_macro_f1"] = int(float(g["macro_f1"]) > float(b["macro_f1"]))
        pair["win_balanced_accuracy"] = int(float(g["balanced_accuracy"]) > float(b["balanced_accuracy"]))
        pairs.append(pair)

    if not pairs:
        raise RuntimeError(f"no paired rows for {args.group} vs {args.baseline} in {args.output_tag}")

    by_metric = {f"delta_{metric}": float(mean(float(p[f"delta_{metric}"]) for p in pairs)) for metric in METRIC_KEYS}

    subject_wins = []
    for target in sorted({int(p["target_subject"]) for p in pairs}):
        vals = [int(p["win_macro_f1"]) for p in pairs if int(p["target_subject"]) == target]
        subject_wins.append(float(mean(vals)) > 0.5)
    seed_wins = []
    for seed in sorted({int(p["seed"]) for p in pairs}):
        vals = [int(p["win_macro_f1"]) for p in pairs if int(p["seed"]) == seed]
        seed_wins.append(float(mean(vals)) > 0.5)

    summary = {
        "status": "completed",
        "output_tag": args.output_tag,
        "comparison": f"{args.group}-vs-{args.baseline}",
        "n_pairs": len(pairs),
        **by_metric,
        "subject_win_rate_macro_f1": float(mean(subject_wins)),
        "seed_win_rate_macro_f1": float(mean(seed_wins)),
    }
    return pairs, summary


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Group Comparison Summary",
        "",
        f"- Output tag: `{summary['output_tag']}`",
        f"- Comparison: `{summary['comparison']}`",
        f"- Pairs: `{summary['n_pairs']}`",
        "",
        "## Deltas",
        "",
    ]
    for key in [f"delta_{m}" for m in METRIC_KEYS]:
        lines.append(f"- `{key}`: `{summary[key]:.6f}`")
    lines.extend(
        [
            f"- `subject_win_rate_macro_f1`: `{summary['subject_win_rate_macro_f1']:.6f}`",
            f"- `seed_win_rate_macro_f1`: `{summary['seed_win_rate_macro_f1']:.6f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--write-tag", default=None)
    args = parser.parse_args()
    if args.write_tag is None:
        args.write_tag = f"{args.output_tag}_{args.group}_vs_{args.baseline}".replace("/", "_")
    return args


def main() -> None:
    args = parse_args()
    out = OUT_DIR / "group_comparisons"
    out.mkdir(parents=True, exist_ok=True)
    pairs, summary = summarize(args)
    write_csv(out / f"group_pairs_{args.write_tag}.csv", pairs)
    (out / f"group_summary_{args.write_tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out / f"GROUP_COMPARISON_{args.write_tag}.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
