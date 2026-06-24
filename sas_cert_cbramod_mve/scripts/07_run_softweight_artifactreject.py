from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Run SoftWeight / ArtifactReject shadow validation from cached CBraMod MVE features.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--bcic2a_root", default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="20,21,22")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--reject_fraction", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.training.linear_probe import LinearHead, clone_head, evaluate_head, train_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    shadow_dir = ensure_dir(outputs / "softweight_artifactreject")
    feature_file = outputs / "features" / "original_features.npz"
    cert_file = outputs / "layer2" / "cert_scores.csv"
    protocol_file = outputs / "audit" / "protocol_audit.json"
    if not feature_file.exists() or not cert_file.exists():
        raise FileNotFoundError("Missing original feature cache or cert_scores.csv. Run full MVE first.")

    with protocol_file.open() as handle:
        protocol_audit = json.load(handle)
    if protocol_audit.get("protocol_leakage_detected"):
        raise RuntimeError("Refusing shadow validation because protocol leakage was detected in the main MVE.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    subjects = [int(v) for v in args.subjects.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    cache = np.load(feature_file, allow_pickle=False)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    trial_ids = cache["trial_ids"].astype(str)
    id_to_idx = {tid: i for i, tid in enumerate(trial_ids.tolist())}
    records = load_bcic2a_trials(str(bcic2a_root), subjects=subjects)

    cert_by_fold = defaultdict(list)
    with cert_file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["target_subject"] = int(row["target_subject"])
            row["seed"] = int(row["seed"])
            row["is_bad"] = int(row["is_bad"])
            row["label"] = int(row["label"])
            for key in ["sas_score", "artifact_risk_raw", "content_score", "style_score", "physio_score", "artifact_safe_score"]:
                row[key] = float(row[key])
            cert_by_fold[(row["target_subject"], row["seed"])].append(row)

    metrics_rows: List[Dict[str, object]] = []
    for target in subjects:
        for seed in seeds:
            print(f"[shadow] target={target} seed={seed}", flush=True)
            set_seed(seed)
            source_records, support_records, test_records = make_few_shot_split(records, target, args.shot, seed)
            source_idx = np.asarray([id_to_idx[r.trial_id] for r in source_records])
            support_idx = np.asarray([id_to_idx[r.trial_id] for r in support_records])
            test_idx = np.asarray([id_to_idx[r.trial_id] for r in test_records])
            source_x, source_y = features[source_idx], labels[source_idx]
            support_x, support_y = features[support_idx], labels[support_idx]
            test_x, test_y = features[test_idx], labels[test_idx]

            fold_scores = cert_by_fold[(target, seed)]
            fold_cache = np.load(outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz", allow_pickle=False)
            cand_x = fold_cache["features"].astype(np.float32)
            cand_y = fold_cache["labels"].astype(np.int64)
            if len(fold_scores) != len(cand_y):
                raise RuntimeError(f"Fold target={target} seed={seed} score/cache length mismatch.")

            score = np.asarray([r["sas_score"] for r in fold_scores], dtype=float)
            artifact_risk = np.asarray([r["artifact_risk_raw"] for r in fold_scores], dtype=float)
            top_half = np.argsort(score)[::-1][: max(1, len(score) // 2)]
            random_half = np.random.default_rng(seed + target * 3000).choice(len(score), size=max(1, len(score) // 2), replace=False)
            reject_n = max(1, int(round(len(score) * args.reject_fraction)))
            artifact_keep = np.ones(len(score), dtype=bool)
            artifact_keep[np.argsort(artifact_risk)[::-1][:reject_n]] = False
            soft_w = 0.2 + 0.8 * ranknorm(score)

            base_head = LinearHead(200, 4)
            train_head(base_head, source_x, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)

            groups = {
                "NoAug": (support_x, support_y, None),
                "NaiveAug": (np.concatenate([support_x, cand_x]), np.concatenate([support_y, cand_y]), None),
                "Random50": (np.concatenate([support_x, cand_x[random_half]]), np.concatenate([support_y, cand_y[random_half]]), None),
                "SASCertTop50": (np.concatenate([support_x, cand_x[top_half]]), np.concatenate([support_y, cand_y[top_half]]), None),
                "SASCertSoftWeight": (
                    np.concatenate([support_x, cand_x]),
                    np.concatenate([support_y, cand_y]),
                    np.concatenate([np.ones(len(support_y), dtype=np.float32), soft_w.astype(np.float32)]),
                ),
                "ArtifactReject": (
                    np.concatenate([support_x, cand_x[artifact_keep]]),
                    np.concatenate([support_y, cand_y[artifact_keep]]),
                    None,
                ),
                "SoftWeightArtifactReject": (
                    np.concatenate([support_x, cand_x[artifact_keep]]),
                    np.concatenate([support_y, cand_y[artifact_keep]]),
                    np.concatenate([np.ones(len(support_y), dtype=np.float32), soft_w[artifact_keep].astype(np.float32)]),
                ),
            }

            for group, (train_x, train_y, weights) in groups.items():
                head = clone_head(base_head)
                train_head(head, train_x, train_y, device, args.finetune_epochs, args.batch_size, args.finetune_lr, args.weight_decay, sample_weight=weights)
                metrics = evaluate_head(head, test_x, test_y, device, args.batch_size)
                metrics.update(
                    {
                        "target_subject": target,
                        "seed": seed,
                        "group": group,
                        "n_train": int(len(train_y)),
                        "shadow_diagnostic": int(group in {"SASCertSoftWeight", "ArtifactReject", "SoftWeightArtifactReject"}),
                    }
                )
                metrics_rows.append(metrics)

    summary, paired, by_subject = summarize(metrics_rows)
    decision = decide(summary, paired)
    payload = {
        "status": "completed",
        "phase": "softweight_artifactreject_shadow_validation",
        "groups": sorted(set(r["group"] for r in metrics_rows)),
        "summary": summary,
        "paired_comparison": paired,
        "decision": decision,
        "protocol_leakage_detected": False,
        "source": "cached_original_and_augmented_cbramod_features_from_main_mve",
    }
    write_csv(shadow_dir / "shadow_layer3_metrics.csv", metrics_rows)
    write_csv(shadow_dir / "shadow_paired_comparison.csv", paired)
    write_csv(shadow_dir / "shadow_by_subject.csv", by_subject)
    write_json(shadow_dir / "shadow_summary.json", payload)
    write_report(shadow_dir / "SOFTWEIGHT_ARTIFACTREJECT_REPORT.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def ranknorm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True) if len(values) > 1 else 1.0
    return ranks.astype(np.float32)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows: Sequence[Dict[str, object]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary = []
    for group, vals in sorted(grouped.items()):
        item = {"group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll"]:
            item[metric] = mean([v[metric] for v in vals])
        summary.append(item)

    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["target_subject"], row["seed"])][row["group"]] = row
    paired = []
    target_groups = ["SASCertSoftWeight", "ArtifactReject", "SoftWeightArtifactReject", "SASCertTop50"]
    baselines = ["NaiveAug", "Random50", "NoAug", "SASCertTop50"]
    for group in target_groups:
        for baseline in baselines:
            if group == baseline:
                continue
            for metric in ["acc", "macro_f1", "ece", "nll"]:
                diffs = []
                for vals in by_key.values():
                    if group in vals and baseline in vals:
                        diffs.append(float(vals[group][metric]) - float(vals[baseline][metric]))
                paired.append(
                    {
                        "comparison": f"{group}_minus_{baseline}",
                        "metric": metric,
                        "mean_delta": mean(diffs),
                        "positive_folds": int(np.sum(np.asarray(diffs) > 0)) if diffs else 0,
                        "n": len(diffs),
                    }
                )

    by_subject_group = defaultdict(list)
    for row in rows:
        by_subject_group[(row["target_subject"], row["group"])].append(row)
    by_subject = []
    for (subject, group), vals in sorted(by_subject_group.items()):
        item = {"target_subject": subject, "group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll"]:
            item[metric] = mean([v[metric] for v in vals])
        by_subject.append(item)
    return summary, paired, by_subject


def paired_delta(paired: Sequence[Dict[str, object]], comparison: str, metric: str) -> float:
    for row in paired:
        if row["comparison"] == comparison and row["metric"] == metric:
            return float(row["mean_delta"])
    return float("nan")


def decide(summary, paired):
    candidates = ["SASCertSoftWeight", "ArtifactReject", "SoftWeightArtifactReject"]
    decisions = {}
    for group in candidates:
        acc_vs_naive = paired_delta(paired, f"{group}_minus_NaiveAug", "acc")
        f1_vs_naive = paired_delta(paired, f"{group}_minus_NaiveAug", "macro_f1")
        ece_vs_naive = paired_delta(paired, f"{group}_minus_NaiveAug", "ece")
        acc_vs_top50 = paired_delta(paired, f"{group}_minus_SASCertTop50", "acc")
        decisions[group] = {
            "acc_vs_naive": acc_vs_naive,
            "macro_f1_vs_naive": f1_vs_naive,
            "ece_vs_naive": ece_vs_naive,
            "acc_vs_top50": acc_vs_top50,
            "passes_shadow_gate": (acc_vs_naive >= 0.005 or f1_vs_naive >= 0.005) and ece_vs_naive <= 0.01,
        }
    best = max(decisions, key=lambda k: decisions[k]["acc_vs_naive"])
    passing = [group for group in candidates if decisions[group]["passes_shadow_gate"]]
    best_passing = max(passing, key=lambda k: decisions[k]["acc_vs_naive"]) if passing else None
    return {
        "best_group_by_acc_vs_naive": best,
        "best_passing_shadow_group": best_passing,
        "group_decisions": decisions,
        "recommended_next": "PROMOTE_SOFTWEIGHT_SHADOW_TO_MAIN" if best_passing else "REFINE_CERT_FEATURES_BEFORE_MORE_TRAINING",
    }


def write_report(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# SoftWeight / ArtifactReject Shadow Validation",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
