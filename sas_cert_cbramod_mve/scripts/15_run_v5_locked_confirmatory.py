from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def load_v4_helpers(project_root: Path):
    path = project_root / "scripts" / "12_run_v4_confirmatory_mve.py"
    spec = importlib.util.spec_from_file_location("v4_confirm_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run v5 locked-method confirmatory MVE.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--bcic2a_root", default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
    parser.add_argument("--main_method", default="SASCert_SoftAR_LS010")
    parser.add_argument("--score_variant", default="artifact_gate_content_rank")
    parser.add_argument("--artifact_reject_percentile", type=float, default=10.0)
    parser.add_argument("--w_min", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 35)))
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)
    h = load_v4_helpers(project_root)

    from sas_cert.augmentations.ops import AugmentedSample, augment_one
    from sas_cert.backbones.cbramod_wrapper import CBraModFeatureExtractor
    from sas_cert.cert.sas_cert_lite import score_candidates
    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.training.linear_probe import LinearHead, clone_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "layer3_v5")
    audit_dir = ensure_dir(outputs / "v5_audit")
    audit = h.file_audit(project_root, outputs)
    write_json(audit_dir / "locked_confirmatory_file_audit.json", audit)
    if audit["blocked"]:
        h.write_blocked(outputs, audit["blocker"])
        raise RuntimeError(audit["blocker"])
    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected"):
        h.write_blocked(outputs, "Existing protocol audit reports leakage.")
        raise RuntimeError("Existing protocol audit reports leakage.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    original = np.load(outputs / "features" / "original_features.npz", allow_pickle=False)
    features = original["features"].astype(np.float32)
    labels = original["labels"].astype(np.int64)
    trial_ids = original["trial_ids"].astype(str)
    id_to_idx = {tid: i for i, tid in enumerate(trial_ids.tolist())}
    records = load_bcic2a_trials(str(bcic2a_root), subjects=args.subjects)
    cert_by_fold = h.load_existing_cert(outputs / "layer2" / "cert_scores.csv")

    needs_extractor = any(
        not (outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz").exists() or (target, seed) not in cert_by_fold
        for target in args.subjects
        for seed in args.seeds
    )
    extractor = CBraModFeatureExtractor(str(project_root / "third_party" / "CBraMod"), device=device) if needs_extractor else None
    metrics_rows: List[Dict[str, object]] = []
    reject_fraction = args.artifact_reject_percentile / 100.0

    for target in args.subjects:
        for seed in args.seeds:
            print(f"[v5-locked] target={target} seed={seed}", flush=True)
            set_seed(seed)
            source_records, support_records, test_records = make_few_shot_split(records, target, args.shot, seed)
            source_idx = np.asarray([id_to_idx[r.trial_id] for r in source_records])
            support_idx = np.asarray([id_to_idx[r.trial_id] for r in support_records])
            test_idx = np.asarray([id_to_idx[r.trial_id] for r in test_records])
            source_x_feat, source_y = features[source_idx], labels[source_idx]
            support_x_feat, support_y = features[support_idx], labels[support_idx]
            test_x_feat, test_y = features[test_idx], labels[test_idx]
            support_raw = np.stack([r.x for r in support_records]).astype(np.float32)
            source_raw = np.stack([r.x for r in source_records]).astype(np.float32)

            cand_feat, cand_y, fold_scores = h.get_candidate_pool(
                outputs,
                cert_by_fold,
                target,
                seed,
                support_raw,
                support_y,
                support_x_feat,
                source_raw,
                args,
                extractor,
                device,
                project_root,
                score_candidates,
                AugmentedSample,
                augment_one,
            )
            score = h.compute_variant_scores(fold_scores, args.score_variant)
            artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in fold_scores], dtype=float)
            keep = np.ones(len(score), dtype=bool)
            keep[np.argsort(artifact_risk)[::-1][: max(1, int(round(len(score) * reject_fraction)))]] = False
            sw = args.w_min + (1.0 - args.w_min) * h.ranknorm(score)

            base_head = LinearHead(200, 4)
            h.train_head(base_head, source_x_feat, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)
            groups = {
                "NaiveAug_raw": (np.concatenate([support_x_feat, cand_feat]), np.concatenate([support_y, cand_y]), None, 0.0),
                "NaiveAug_LS010": (np.concatenate([support_x_feat, cand_feat]), np.concatenate([support_y, cand_y]), None, args.label_smoothing),
                "SASCert_SoftAR_LS010": (
                    np.concatenate([support_x_feat, cand_feat[keep]]),
                    np.concatenate([support_y, cand_y[keep]]),
                    np.concatenate([np.ones(len(support_y), dtype=np.float32), sw[keep].astype(np.float32)]),
                    args.label_smoothing,
                ),
                "ArtifactReject_raw_diagnostic": (
                    np.concatenate([support_x_feat, cand_feat[keep]]),
                    np.concatenate([support_y, cand_y[keep]]),
                    None,
                    args.label_smoothing,
                ),
            }
            for group, (train_x, train_y, weights, eps) in groups.items():
                head = clone_head(base_head)
                h.train_head(head, train_x, train_y, device, args.finetune_epochs, args.batch_size, args.finetune_lr, args.weight_decay, sample_weight=weights, label_smoothing=eps)
                row = h.evaluate(head, test_x_feat, test_y, device, args.batch_size)
                row.update(
                    {
                        "target_subject": target,
                        "seed": seed,
                        "group": group,
                        "n_train": int(len(train_y)),
                        "score_variant": args.score_variant,
                        "artifact_reject_percentile": args.artifact_reject_percentile,
                        "w_min": args.w_min,
                        "label_smoothing": eps,
                    }
                )
                metrics_rows.append(row)

    h.add_group_stability(metrics_rows)
    summary, paired, by_subject, win_rate, seed_table = summarize_v5(metrics_rows)
    payload = {
        "status": "completed",
        "main_method": args.main_method,
        "score_variant": args.score_variant,
        "artifact_reject_percentile": args.artifact_reject_percentile,
        "w_min": args.w_min,
        "label_smoothing": args.label_smoothing,
        "seeds": args.seeds,
        "summary": summary,
        "paired_comparison": paired,
        "protocol_leakage_detected": False,
        "backbone_frozen": True,
    }
    write_csv(out_dir / "layer3_v5_metrics.csv", metrics_rows)
    write_csv(out_dir / "paired_comparison_v5.csv", paired)
    write_csv(out_dir / "by_subject_table_v5.csv", by_subject)
    write_csv(out_dir / "win_rate_table_v5.csv", win_rate)
    write_csv(out_dir / "seed_stability_table_v5.csv", seed_table)
    write_json(out_dir / "layer3_v5_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def mean(vals: Iterable[float]) -> float:
    vals = [float(v) for v in vals]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_v5(rows: Sequence[Dict[str, object]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary = []
    for group, vals in sorted(grouped.items()):
        item = {"group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "mean_confidence"]:
            item[metric] = mean(v[metric] for v in vals)
        item["worst_subject_acc"] = vals[0]["worst_subject_acc"]
        item["subject_wise_std"] = vals[0]["subject_wise_std"]
        summary.append(item)

    by_subject = aggregate(rows, "target_subject")
    seed_table = aggregate(rows, "seed")
    win_rate = []
    paired = []
    for baseline in ["NaiveAug_raw", "NaiveAug_LS010"]:
        base = next(row for row in summary if row["group"] == baseline)
        for group in ["SASCert_SoftAR_LS010", "ArtifactReject_raw_diagnostic"]:
            row = next(r for r in summary if r["group"] == group)
            delta = {m: row[m] - base[m] for m in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "worst_subject_acc", "subject_wise_std"]}
            subject_win_acc, subject_win_f1 = win_rates(by_subject, group, baseline, "target_subject")
            seed_win_acc, seed_win_f1 = win_rates(seed_table, group, baseline, "seed")
            passed = {
                "passed_macro_f1": delta["macro_f1"] >= (0.01 if baseline == "NaiveAug_raw" else 0.005),
                "passed_acc": delta["acc"] >= (0.005 if baseline == "NaiveAug_raw" else 0.0),
                "passed_ece": delta["ece"] <= (0.01 if baseline == "NaiveAug_raw" else 0.005),
                "passed_nll": delta["nll"] <= 0.005,
                "passed_brier": delta["brier"] <= 0.005,
            }
            if group == "SASCert_SoftAR_LS010":
                if baseline == "NaiveAug_raw":
                    decision = (
                        "GO_RAW_BASELINE"
                        if all(passed.values()) and subject_win_f1 >= 0.75 and subject_win_acc >= 0.70 and seed_win_f1 >= 0.70 and seed_win_acc >= 0.65
                        else classify_failure(delta, passed)
                    )
                else:
                    decision = "GO_LS_BASELINE" if all(passed.values()) else classify_failure(delta, passed)
            else:
                decision = "ARTIFACT_ONLY_BRANCH_DIAGNOSTIC"
            paired.append(
                {
                    "group": group,
                    "baseline_group": baseline,
                    **{f"delta_{k}": v for k, v in delta.items()},
                    "subject_win_rate_acc": subject_win_acc,
                    "subject_win_rate_macro_f1": subject_win_f1,
                    "seed_win_rate_acc": seed_win_acc,
                    "seed_win_rate_macro_f1": seed_win_f1,
                    **passed,
                    "final_group_decision": decision,
                }
            )
            win_rate.append(
                {
                    "group": group,
                    "baseline_group": baseline,
                    "subject_win_rate_acc": subject_win_acc,
                    "subject_win_rate_macro_f1": subject_win_f1,
                    "seed_win_rate_acc": seed_win_acc,
                    "seed_win_rate_macro_f1": seed_win_f1,
                }
            )
    return summary, paired, by_subject, win_rate, seed_table


def classify_failure(delta: Dict[str, float], passed: Dict[str, bool]) -> str:
    if not passed["passed_ece"]:
        return "CALIBRATION_REGRESSION"
    if passed["passed_macro_f1"] and not passed["passed_acc"]:
        return "PARTIAL_GO_F1_STABLE"
    return "EFFECT_NOT_STABLE"


def aggregate(rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row[key], row["group"])].append(row)
    out = []
    for (key_value, group), vals in sorted(grouped.items()):
        item = {key: key_value, "group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "mean_confidence"]:
            item[metric] = mean(v[metric] for v in vals)
        out.append(item)
    return out


def win_rates(table: Sequence[Dict[str, object]], group: str, baseline: str, key: str) -> Tuple[float, float]:
    by_key = defaultdict(dict)
    for row in table:
        by_key[row[key]][row["group"]] = row
    acc = []
    f1 = []
    for vals in by_key.values():
        if group in vals and baseline in vals:
            acc.append(float(vals[group]["acc"]) > float(vals[baseline]["acc"]))
            f1.append(float(vals[group]["macro_f1"]) > float(vals[baseline]["macro_f1"]))
    return mean(acc), mean(f1)


if __name__ == "__main__":
    main()
