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
    parser = argparse.ArgumentParser(description="Run SAS-Cert-CBraMod v2 SoftWeightArtifactReject training.")
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
    parser.add_argument("--score_variant", default="best_non_oracle")
    parser.add_argument("--artifact_reject_percentile", type=float, default=10.0)
    parser.add_argument("--w_min", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.training.linear_probe import LinearHead, clone_head, evaluate_head, train_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "layer3_v2")
    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected"):
        raise RuntimeError("Refusing v2 training because protocol leakage was detected.")

    v2_summary_path = outputs / "layer2_v2" / "layer2_v2_summary.json"
    if not v2_summary_path.exists():
        raise FileNotFoundError("Missing layer2_v2_summary.json. Run scripts/07_diagnose_cert_v2.py first.")
    v2_summary = json.loads(v2_summary_path.read_text())
    selected_variant = v2_summary["best_non_oracle_variant"] if args.score_variant == "best_non_oracle" else args.score_variant

    feature_file = outputs / "features" / "original_features.npz"
    cert_file = outputs / "layer2" / "cert_scores.csv"
    if not feature_file.exists() or not cert_file.exists():
        raise FileNotFoundError("Missing original features or cert scores. Run main MVE first.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    subjects = [int(v) for v in args.subjects.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    original = np.load(feature_file, allow_pickle=False)
    features = original["features"].astype(np.float32)
    labels = original["labels"].astype(np.int64)
    trial_ids = original["trial_ids"].astype(str)
    id_to_idx = {tid: idx for idx, tid in enumerate(trial_ids.tolist())}

    records = load_bcic2a_trials(str(bcic2a_root), subjects=subjects)
    cert_by_fold = load_cert_by_fold(cert_file)
    metrics_rows: List[Dict[str, object]] = []

    reject_fraction = args.artifact_reject_percentile / 100.0
    for target in subjects:
        for seed in seeds:
            print(f"[layer3_v2] target={target} seed={seed} variant={selected_variant}", flush=True)
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

            score = compute_variant_scores(fold_scores, selected_variant, bool(v2_summary["physio_fixed_used"]))
            hard_top50_score = np.asarray([float(r["sas_score"]) for r in fold_scores], dtype=float)
            artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in fold_scores], dtype=float)

            random_half = np.random.default_rng(seed + target * 3000).choice(len(score), size=max(1, len(score) // 2), replace=False)
            top_half = np.argsort(hard_top50_score)[::-1][: max(1, len(score) // 2)]
            reject_n = max(1, int(round(len(score) * reject_fraction)))
            artifact_keep = np.ones(len(score), dtype=bool)
            artifact_keep[np.argsort(artifact_risk)[::-1][:reject_n]] = False
            soft_w = args.w_min + (1.0 - args.w_min) * ranknorm(score)

            base_head = LinearHead(200, 4)
            train_head(base_head, source_x, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)

            groups = {
                "NoAug": (support_x, support_y, None),
                "NaiveAug": (np.concatenate([support_x, cand_x]), np.concatenate([support_y, cand_y]), None),
                "Random50": (np.concatenate([support_x, cand_x[random_half]]), np.concatenate([support_y, cand_y[random_half]]), None),
                "SASCertTop50": (np.concatenate([support_x, cand_x[top_half]]), np.concatenate([support_y, cand_y[top_half]]), None),
                "ArtifactReject": (np.concatenate([support_x, cand_x[artifact_keep]]), np.concatenate([support_y, cand_y[artifact_keep]]), None),
                "SoftWeight": (
                    np.concatenate([support_x, cand_x]),
                    np.concatenate([support_y, cand_y]),
                    np.concatenate([np.ones(len(support_y), dtype=np.float32), soft_w.astype(np.float32)]),
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
                        "score_variant": selected_variant,
                        "artifact_reject_percentile": args.artifact_reject_percentile,
                        "w_min": args.w_min,
                        "layer": "layer3_v2",
                    }
                )
                metrics_rows.append(metrics)

    summary, paired, by_subject = summarize(metrics_rows)
    decision = decide(paired)
    payload = {
        "status": "completed",
        "selected_score_variant": selected_variant,
        "requested_score_variant": args.score_variant,
        "physio_fixed_used": bool(v2_summary["physio_fixed_used"]),
        "artifact_reject_percentile": args.artifact_reject_percentile,
        "w_min": args.w_min,
        "summary": summary,
        "paired_comparison": paired,
        "decision": decision,
        "protocol_leakage_detected": False,
        "backbone_frozen": True,
    }
    write_csv(out_dir / "layer3_v2_metrics.csv", metrics_rows)
    write_csv(out_dir / "paired_comparison_v2.csv", paired)
    write_csv(out_dir / "by_subject_table_v2.csv", by_subject)
    write_json(out_dir / "layer3_v2_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def load_cert_by_fold(path: Path) -> Dict[tuple, List[Dict[str, object]]]:
    out: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    numeric = ["sas_score", "artifact_risk_raw", "artifact_safe_score", "content_score", "style_score", "physio_score"]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["target_subject"] = int(row["target_subject"])
            row["seed"] = int(row["seed"])
            row["is_bad"] = int(row["is_bad"])
            row["label"] = int(row["label"])
            for key in numeric:
                row[key] = float(row[key])
            out[(row["target_subject"], row["seed"])].append(row)
    return out


def ranknorm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks.astype(np.float32)


def compute_variant_scores(rows: Sequence[Dict[str, object]], variant: str, physio_fixed_used: bool) -> np.ndarray:
    artifact = ranknorm([float(r["artifact_safe_score"]) for r in rows])
    content = ranknorm([float(r["content_score"]) for r in rows])
    style = ranknorm([float(r["style_score"]) for r in rows])
    physio = ranknorm([float(r["physio_score"]) for r in rows])
    physio_fixed = 1.0 - physio if physio_fixed_used else physio
    artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in rows], dtype=float)
    if variant == "current_total":
        return 0.35 * content + 0.25 * physio + 0.25 * artifact + 0.15 * style
    if variant == "artifact_content":
        return 0.55 * artifact + 0.45 * content
    if variant == "artifact_content_style":
        return 0.45 * artifact + 0.40 * content + 0.15 * style
    if variant == "artifact_content_physio_fixed":
        return 0.40 * artifact + 0.35 * content + 0.15 * physio_fixed + 0.10 * style
    if variant == "artifact_gate_content_rank":
        score = content.copy()
        reject_n = max(1, int(round(len(score) * 0.10)))
        score[np.argsort(artifact_risk)[::-1][:reject_n]] = 0.0
        return score
    raise ValueError(f"Unknown score variant: {variant}")


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
    targets = ["SoftWeightArtifactReject", "SoftWeight", "ArtifactReject", "SASCertTop50"]
    baselines = ["NaiveAug", "Random50", "SASCertTop50", "NoAug"]
    for group in targets:
        for baseline in baselines:
            if group == baseline:
                continue
            for metric in ["acc", "macro_f1", "ece", "nll"]:
                diffs = [float(vals[group][metric]) - float(vals[baseline][metric]) for vals in by_key.values() if group in vals and baseline in vals]
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


def decide(paired: Sequence[Dict[str, object]]) -> Dict[str, object]:
    acc = paired_delta(paired, "SoftWeightArtifactReject_minus_NaiveAug", "acc")
    f1 = paired_delta(paired, "SoftWeightArtifactReject_minus_NaiveAug", "macro_f1")
    ece = paired_delta(paired, "SoftWeightArtifactReject_minus_NaiveAug", "ece")
    return {
        "softweight_artifactreject_minus_naive_acc": acc,
        "softweight_artifactreject_minus_naive_macro_f1": f1,
        "softweight_artifactreject_minus_naive_ece": ece,
        "go_softweight": (acc >= 0.01 or f1 >= 0.01) and ece <= 0.01,
    }


if __name__ == "__main__":
    main()
