from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def bootstrap(project_root: str):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Run full SAS-Cert-CBraMod MVE.")
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--workspace_root", required=True)
    parser.add_argument("--cbramod_src", required=True)
    parser.add_argument("--bcic2a_root", required=True)
    parser.add_argument("--old_eegnet_report", required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--continue_on_nogo", default="true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="20,21,22")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.augmentations import augmentation_sanity, generate_augmented_set
    from sas_cert.augmentations.ops import AugmentedSample, augment_one
    from sas_cert.backbones.cbramod_wrapper import CBraModFeatureExtractor
    from sas_cert.cert.sas_cert_lite import score_candidates
    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split, records_to_arrays
    from sas_cert.metrics.classification import roc_auc_binary
    from sas_cert.training.linear_probe import LinearHead, clone_head, evaluate_head, train_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    audit_dir = ensure_dir(outputs / "audit")
    ensure_dir(outputs / "smoke")
    ensure_dir(outputs / "features" / "augmented")
    ensure_dir(outputs / "layer1")
    ensure_dir(outputs / "layer2")
    ensure_dir(outputs / "layer3")

    def block(reason: str):
        payload = {"status": "blocked", "blocker": reason}
        write_json(outputs / "BLOCKED_REPORT.md", payload)
        raise RuntimeError(reason)

    workspace_root = Path(args.workspace_root).resolve()
    cbramod_src = (workspace_root / args.cbramod_src).resolve() if not Path(args.cbramod_src).is_absolute() else Path(args.cbramod_src)
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    old_report = Path(args.old_eegnet_report).resolve()

    path_audit = {
        "workspace_root": str(workspace_root),
        "project_root": str(project_root),
        "cbramod_src": str(cbramod_src),
        "bcic2a_root": str(bcic2a_root),
        "old_eegnet_report": str(old_report),
        "workspace_exists": workspace_root.exists(),
        "cbramod_src_exists": cbramod_src.exists(),
        "cbramod_weight_exists": (cbramod_src / "pretrained_weights" / "pretrained_weights.pth").exists(),
        "bcic2a_root_exists": bcic2a_root.exists(),
        "bcic2a_mat_count": len(list(bcic2a_root.glob("A0*.mat"))) if bcic2a_root.exists() else 0,
        "data_copy_detected": len(list(project_root.glob("**/A0*.mat"))) > 0,
        "old_outputs_used_as_input": False,
    }
    write_json(audit_dir / "path_audit.json", path_audit)
    if not path_audit["workspace_exists"] or not path_audit["cbramod_src_exists"] or not path_audit["cbramod_weight_exists"]:
        block("Workspace or CBraMod source/weights missing.")
    if not path_audit["bcic2a_root_exists"] or path_audit["bcic2a_mat_count"] != 18:
        block("BCIC2a root missing or .mat count is not 18.")
    if path_audit["data_copy_detected"]:
        block("Raw BCIC2a .mat files detected inside new project.")

    # Project setup audit. The selective copy is already done by the outer implementation,
    # but this keeps the one-command runner robust if it is invoked later.
    third_party = project_root / "third_party" / "CBraMod"
    if not (third_party / "models" / "cbramod.py").exists():
        ensure_dir(third_party)
        shutil.copytree(cbramod_src / "models", third_party / "models", dirs_exist_ok=True)
        shutil.copytree(cbramod_src / "pretrained_weights", third_party / "pretrained_weights", dirs_exist_ok=True)
        shutil.copy2(cbramod_src / "requirements.txt", third_party / "requirements.txt")
    setup_audit = {
        "third_party_cbramod": str(third_party),
        "models_exists": (third_party / "models" / "cbramod.py").exists(),
        "weights_exists": (third_party / "pretrained_weights" / "pretrained_weights.pth").exists(),
        "copied_data_detected": len(list(third_party.glob("**/A0*.mat"))) > 0,
        "copy_mode": "selective_models_pretrained_weights_requirements",
    }
    write_json(audit_dir / "project_setup.json", setup_audit)
    if not setup_audit["models_exists"] or not setup_audit["weights_exists"] or setup_audit["copied_data_detected"]:
        block("Project setup failed or copied raw data.")

    subjects = [int(v) for v in args.subjects.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    if args.dry_run:
        subjects = [1, 3, 7]
        seeds = [20]

    records = load_bcic2a_trials(str(bcic2a_root), subjects=subjects)
    x_all, y_all, subjects_all, sessions_all, trial_ids_all = records_to_arrays(records)
    if x_all.shape[1:] != (22, 800) or sorted(set(y_all.tolist())) != [0, 1, 2, 3]:
        block("Loaded data failed shape or label audit.")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    extractor = CBraModFeatureExtractor(str(third_party), device=device)
    smoke_features, smoke_reps = extractor.extract(x_all[: min(8, len(x_all))], batch_size=min(8, args.feature_batch_size))
    smoke = {
        "passed": bool(smoke_features.shape[1:] == (200,) and not np.isnan(smoke_features).any()),
        "loaded_trials": int(len(x_all)),
        "loaded_subjects": sorted(set(subjects_all.tolist())),
        "sessions": sorted(set(sessions_all.tolist())),
        "raw_shape_tail": list(x_all.shape[1:]),
        "cbramod_input_shape_tail": [22, 4, 200],
        "cbramod_output_shape": list(smoke_reps.shape),
        "pooled_feature_shape": list(smoke_features.shape),
        "feature_nan_count": int(np.isnan(smoke_features).sum()),
        "feature_inf_count": int(np.isinf(smoke_features).sum()),
        "checkpoint_audit": extractor.checkpoint_audit,
    }
    write_json(outputs / "smoke" / "smoke_report.json", smoke)
    (outputs / "smoke" / "smoke_log.txt").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
    if not smoke["passed"] or extractor.checkpoint_audit["unexpected_keys"] != 0:
        block("CBraMod smoke failed.")

    print("[features] extracting original CBraMod features", flush=True)
    all_features, _ = extractor.extract(x_all, batch_size=args.feature_batch_size)
    np.savez_compressed(
        outputs / "features" / "original_features.npz",
        features=all_features.astype(np.float32),
        labels=y_all,
        subjects=subjects_all,
        sessions=sessions_all,
        trial_ids=trial_ids_all,
    )
    write_json(
        outputs / "features" / "original_features_manifest.json",
        {
            "contains_raw_mat_data": False,
            "contains_raw_trial_array": False,
            "contains_cbramod_features": True,
            "contains_augmented_raw_array": False,
            "contains_labels": True,
            "labels_scope": "source_train_and_target_support_only_for_training; target_test_only_for_eval",
            "target_test_used_for_style_anchor": False,
            "target_test_used_for_cert_threshold": False,
            "feature_shape": list(all_features.shape),
        },
    )
    id_to_idx = {tid: i for i, tid in enumerate(trial_ids_all.tolist())}

    layer1_rows = []
    layer1_sanity_rows = []
    cert_rows = []
    layer3_rows = []
    by_subject_layer3 = []

    conditions = [
        ("NoAug", None),
        ("CleanAug", "clean"),
        ("BadContent", "bad_content"),
        ("BadStyle", "bad_style"),
        ("BadPhysio", "bad_physio"),
        ("BadArtifact", "bad_artifact"),
    ]

    for target in subjects:
        for seed in seeds:
            print(f"[fold] target={target} seed={seed}", flush=True)
            set_seed(seed)
            source_records, support_records, test_records = make_few_shot_split(records, target, args.shot, seed)
            source_idx = np.asarray([id_to_idx[r.trial_id] for r in source_records])
            support_idx = np.asarray([id_to_idx[r.trial_id] for r in support_records])
            test_idx = np.asarray([id_to_idx[r.trial_id] for r in test_records])

            source_feat, source_y = all_features[source_idx], y_all[source_idx]
            support_feat, support_y = all_features[support_idx], y_all[support_idx]
            test_feat, test_y = all_features[test_idx], y_all[test_idx]
            support_x = np.stack([r.x for r in support_records]).astype(np.float32)
            source_x = np.stack([r.x for r in source_records]).astype(np.float32)

            base_head = LinearHead(200, 4)
            train_head(base_head, source_feat, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)

            def finetune_eval(train_feat, train_y, group_name):
                head = clone_head(base_head)
                train_head(head, train_feat, train_y, device, args.finetune_epochs, args.batch_size, args.finetune_lr, args.weight_decay)
                metrics = evaluate_head(head, test_feat, test_y, device, args.batch_size)
                metrics.update({"target_subject": target, "seed": seed, "group": group_name})
                return metrics

            for condition, aug_type in conditions:
                if aug_type is None:
                    metrics = finetune_eval(support_feat, support_y, condition)
                    metrics.update({"condition": condition, "aug_type": "none", "layer": "layer1", "intensity": 0.0})
                    layer1_rows.append(metrics)
                    continue
                rng = np.random.default_rng(seed + target * 1000 + len(condition))
                samples = generate_augmented_set(support_x, support_y, aug_type, args.n_aug, rng, args.intensity, style_bank=source_x)
                sanity = augmentation_sanity(samples)
                sanity.update({"target_subject": target, "seed": seed, "condition": condition, "aug_type": aug_type})
                layer1_sanity_rows.append(sanity)
                if sanity["nan_count"] or sanity["inf_count"]:
                    block(f"Augmentation {condition} produced NaN/Inf.")
                aug_x = np.stack([s.x for s in samples]).astype(np.float32)
                aug_y = np.asarray([s.y for s in samples], dtype=np.int64)
                aug_feat, _ = extractor.extract(aug_x, batch_size=args.feature_batch_size)
                metrics = finetune_eval(np.concatenate([support_feat, aug_feat]), np.concatenate([support_y, aug_y]), condition)
                metrics.update({"condition": condition, "aug_type": aug_type, "layer": "layer1", "intensity": args.intensity})
                layer1_rows.append(metrics)

            # Layer 2 candidate pool: n_aug clean/mild plus one candidate for each bad type.
            rng = np.random.default_rng(seed + target * 2000)
            candidates = []
            for i in range(support_x.shape[0]):
                for k in range(args.n_aug):
                    x_aug = augment_one(support_x[i], "clean", rng, 0.35, source_x)
                    candidates.append(AugmentedSample(x_aug.astype(np.float32), int(support_y[i]), i, "clean", 0.35, False, f"clean_{i:04d}_{k:02d}"))
                for aug_type in ["bad_content", "bad_style", "bad_physio", "bad_artifact"]:
                    x_aug = augment_one(support_x[i], aug_type, rng, args.intensity, source_x)
                    candidates.append(AugmentedSample(x_aug.astype(np.float32), int(support_y[i]), i, aug_type, args.intensity, True, f"{aug_type}_{i:04d}_00"))
            cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
            cand_y = np.asarray([c.y for c in candidates], dtype=np.int64)
            cand_feat, _ = extractor.extract(cand_x, batch_size=args.feature_batch_size)
            fold_cache = outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz"
            np.savez_compressed(
                fold_cache,
                features=cand_feat.astype(np.float32),
                labels=cand_y,
                aug_type=np.asarray([c.aug_type for c in candidates]),
                is_bad=np.asarray([int(c.is_bad) for c in candidates], dtype=np.int64),
                original_index=np.asarray([c.original_index for c in candidates], dtype=np.int64),
            )
            write_json(
                outputs / "features" / "augmented" / f"target{target}_seed{seed}_manifest.json",
                {
                    "contains_raw_mat_data": False,
                    "contains_raw_trial_array": False,
                    "contains_cbramod_features": True,
                    "contains_augmented_raw_array": False,
                    "contains_labels": True,
                    "target_test_used_for_style_anchor": False,
                    "target_test_used_for_cert_threshold": False,
                    "candidate_count": int(len(candidates)),
                },
            )
            fold_scores = score_candidates(candidates, support_x, support_feat, cand_feat)
            for row in fold_scores:
                row.update({"target_subject": target, "seed": seed})
            cert_rows.extend(fold_scores)

            # Layer 3 groups.
            scores_np = np.asarray([r["sas_score"] for r in fold_scores], dtype=float)
            top_n = max(1, len(candidates) // 2)
            top_idx = np.argsort(scores_np)[::-1][:top_n]
            rand_idx = np.random.default_rng(seed + target * 3000).choice(len(candidates), size=top_n, replace=False)
            groups = {
                "NoAug": (support_feat, support_y),
                "NaiveAug": (np.concatenate([support_feat, cand_feat]), np.concatenate([support_y, cand_y])),
                "Random50": (np.concatenate([support_feat, cand_feat[rand_idx]]), np.concatenate([support_y, cand_y[rand_idx]])),
                "SASCertTop50": (np.concatenate([support_feat, cand_feat[top_idx]]), np.concatenate([support_y, cand_y[top_idx]])),
            }
            for group_name, (train_feat, train_y) in groups.items():
                metrics = finetune_eval(train_feat, train_y, group_name)
                metrics.update({"layer": "layer3", "n_train": int(len(train_y))})
                layer3_rows.append(metrics)

    write_csv(outputs / "layer1" / "layer1_metrics.csv", layer1_rows)
    write_csv(outputs / "layer1" / "layer1_aug_sanity.csv", layer1_sanity_rows)
    write_json(outputs / "layer1" / "layer1_summary.json", summarize_layer1(layer1_rows))
    write_csv(outputs / "layer1" / "layer1_by_subject.csv", group_rows(layer1_rows, ["target_subject", "condition"], ["acc", "macro_f1", "kappa", "ece", "nll"]))

    cert_summary, auc_by_subject, auc_by_aug, component_auc, top_bottom = summarize_layer2(cert_rows)
    write_csv(outputs / "layer2" / "cert_scores.csv", cert_rows)
    write_csv(outputs / "layer2" / "cert_auc_by_subject.csv", auc_by_subject)
    write_csv(outputs / "layer2" / "cert_auc_by_aug_type.csv", auc_by_aug)
    write_csv(outputs / "layer2" / "cert_single_component_auc.csv", component_auc)
    write_csv(outputs / "layer2" / "cert_top_bottom_stats.csv", top_bottom)
    write_json(outputs / "layer2" / "layer2_summary.json", cert_summary)

    layer3_summary, paired, by_subject = summarize_layer3(layer3_rows)
    write_csv(outputs / "layer3" / "layer3_metrics.csv", layer3_rows)
    write_csv(outputs / "layer3" / "paired_comparison.csv", paired)
    write_csv(outputs / "layer3" / "by_subject_table.csv", by_subject)
    write_json(outputs / "layer3" / "layer3_summary.json", layer3_summary)

    protocol_audit = {
        "target_test_used_for_style_anchor": False,
        "target_test_used_for_cert_threshold": False,
        "target_test_used_for_rank_normalization": False,
        "old_outputs_used_as_input": False,
        "backbone_frozen": True,
        "head_only_training": True,
        "data_copy_detected": len(list(project_root.glob("**/A0*.mat"))) > 0,
        "protocol_leakage_detected": False,
    }
    write_json(audit_dir / "protocol_audit.json", protocol_audit)
    old_audit = {
        "old_report_exists": old_report.exists(),
        "old_report_path": str(old_report),
        "used_for_training": False,
        "used_for_cache": False,
        "used_for_threshold": False,
        "used_for_report_baseline_only": True,
    }
    write_json(audit_dir / "old_baseline_readonly_audit.json", old_audit)

    compact = build_compact_result(smoke, layer1_rows, cert_summary, layer3_summary, paired, protocol_audit, seeds, args.shot)
    write_json(outputs / "compact_result.json", compact)
    write_report(outputs / "SAS_CERT_CBRAMOD_MVE_REPORT.md", path_audit, smoke, layer1_rows, cert_summary, layer3_summary, paired, compact, old_audit)
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


def mean(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(values)) if values else float("nan")


def group_rows(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, values in grouped.items():
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = len(values)
        for metric in metrics:
            item[metric] = mean([v[metric] for v in values])
        out.append(item)
    return out


def summarize_layer1(rows):
    grouped = group_rows(rows, ["condition"], ["acc", "macro_f1", "kappa", "ece", "nll"])
    by_cond = {row["condition"]: row for row in grouped}
    clean = by_cond.get("CleanAug", {})
    summary = {"grouped_mean": grouped}
    for cond in ["BadArtifact", "BadContent", "BadPhysio", "BadStyle"]:
        if clean and cond in by_cond:
            summary[f"{cond}_delta_acc_vs_clean"] = by_cond[cond]["acc"] - clean["acc"]
            summary[f"{cond}_delta_kappa_vs_clean"] = by_cond[cond]["kappa"] - clean["kappa"]
    summary["passed"] = (
        summary.get("BadArtifact_delta_acc_vs_clean", 0) <= -0.03
        or summary.get("BadArtifact_delta_kappa_vs_clean", 0) <= -0.03
    ) and (
        summary.get("BadContent_delta_acc_vs_clean", 0) <= -0.02
        or summary.get("BadPhysio_delta_acc_vs_clean", 0) <= -0.02
    )
    return summary


def summarize_layer2(rows):
    from sas_cert.metrics.classification import roc_auc_binary

    labels = 1 - np.asarray([int(r["is_bad"]) for r in rows])
    total = np.asarray([float(r["sas_score"]) for r in rows])
    overall_auc = roc_auc_binary(labels, total)
    auc_by_subject = []
    for (target, seed), vals in group_by(rows, ["target_subject", "seed"]).items():
        y = 1 - np.asarray([int(r["is_bad"]) for r in vals])
        s = np.asarray([float(r["sas_score"]) for r in vals])
        auc_by_subject.append({"target_subject": target, "seed": seed, "total_auc": roc_auc_binary(y, s)})
    auc_by_aug = []
    for aug_type in ["bad_artifact", "bad_content", "bad_physio", "bad_style", "all_bad"]:
        vals = [r for r in rows if r["aug_type"] == "clean" or aug_type == "all_bad" and int(r["is_bad"]) == 1 or r["aug_type"] == aug_type]
        if vals:
            y = np.asarray([1 if r["aug_type"] == "clean" else 0 for r in vals])
            s = np.asarray([float(r["sas_score"]) for r in vals])
            auc_by_aug.append({"comparison": f"clean_vs_{aug_type}", "total_auc": roc_auc_binary(y, s), "n": len(vals)})
    component_auc = []
    for score_key in ["content_score", "style_score", "physio_score", "artifact_safe_score", "sas_score"]:
        component_auc.append({"component": score_key, "auc": roc_auc_binary(labels, np.asarray([float(r[score_key]) for r in rows]))})
    sorted_rows = sorted(rows, key=lambda r: float(r["sas_score"]), reverse=True)
    k = max(1, int(round(len(sorted_rows) * 0.3)))
    top = sorted_rows[:k]
    bottom = sorted_rows[-k:]
    top_bottom = [
        {"bucket": "top30", "bad_rate": mean([r["is_bad"] for r in top]), "n": len(top)},
        {"bucket": "bottom30", "bad_rate": mean([r["is_bad"] for r in bottom]), "n": len(bottom)},
    ]
    summary = {
        "overall_auc": overall_auc,
        "top30_bad_rate": top_bottom[0]["bad_rate"],
        "bottom30_bad_rate": top_bottom[1]["bad_rate"],
        "bad_artifact_auc": next((r["total_auc"] for r in auc_by_aug if r["comparison"] == "clean_vs_bad_artifact"), float("nan")),
        "bad_content_auc": next((r["total_auc"] for r in auc_by_aug if r["comparison"] == "clean_vs_bad_content"), float("nan")),
        "bad_physio_auc": next((r["total_auc"] for r in auc_by_aug if r["comparison"] == "clean_vs_bad_physio"), float("nan")),
        "bad_style_auc": next((r["total_auc"] for r in auc_by_aug if r["comparison"] == "clean_vs_bad_style"), float("nan")),
    }
    summary["passed"] = summary["overall_auc"] >= 0.70 and summary["bad_artifact_auc"] >= 0.80 and summary["top30_bad_rate"] <= 0.05 and summary["bottom30_bad_rate"] >= 0.25
    return summary, auc_by_subject, auc_by_aug, component_auc, top_bottom


def summarize_layer3(rows):
    grouped = group_rows(rows, ["group"], ["acc", "macro_f1", "kappa", "ece", "nll"])
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["target_subject"], row["seed"])][row["group"]] = row
    paired = []
    for baseline in ["NaiveAug", "Random50", "NoAug"]:
        for metric in ["acc", "macro_f1", "ece", "nll"]:
            diffs = []
            for vals in by_key.values():
                if "SASCertTop50" in vals and baseline in vals:
                    diffs.append(float(vals["SASCertTop50"][metric]) - float(vals[baseline][metric]))
            paired.append({"comparison": f"SASCertTop50_minus_{baseline}", "metric": metric, "mean_delta": mean(diffs), "positive_folds": int(np.sum(np.asarray(diffs) > 0)), "n": len(diffs)})
    by_subject = group_rows(rows, ["target_subject", "group"], ["acc", "macro_f1", "kappa", "ece", "nll"])
    get_delta = lambda comp, metric: next((r["mean_delta"] for r in paired if r["comparison"] == comp and r["metric"] == metric), float("nan"))
    summary = {"grouped_mean": grouped, "paired_comparison": paired}
    summary["sas_top50_minus_random50_acc"] = get_delta("SASCertTop50_minus_Random50", "acc")
    summary["sas_top50_minus_naive_acc"] = get_delta("SASCertTop50_minus_NaiveAug", "acc")
    summary["sas_top50_minus_naive_ece"] = get_delta("SASCertTop50_minus_NaiveAug", "ece")
    summary["passed"] = summary["sas_top50_minus_random50_acc"] >= 0.01 and summary["sas_top50_minus_naive_acc"] >= 0.0 and summary["sas_top50_minus_naive_ece"] <= 0.01
    return summary, paired, by_subject


def group_by(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def build_compact_result(smoke, l1_rows, l2, l3, paired, protocol_audit, seeds, shot):
    l1 = summarize_layer1(l1_rows)
    interpretation = "CONFIRMATORY" if l2.get("passed") else "DIAGNOSTIC_ONLY"
    if l2.get("passed") and l3.get("passed"):
        decision = "GO"
        next_action = "Run longer confirmatory CBraMod MVE or pre-register SoftWeight comparison."
    elif l2.get("overall_auc", 0) >= 0.60:
        decision = "SOFTWEIGHT_NEXT" if l3.get("sas_top50_minus_naive_acc", -1) < 0.005 else "REFINE_CERT"
        next_action = "Refine artifact/content/physio certificate and run SoftWeight/ArtifactReject shadow groups."
    else:
        decision = "STOP"
        next_action = "Do not expand training; rebuild certificate items first."
    return {
        "project": "sas_cert_cbramod_mve",
        "status": "completed",
        "protocol": "FEWSHOT_TARGET_SUPPORT",
        "backbone": "CBraMod",
        "backbone_frozen": True,
        "dataset": "BCIC-IV-2a",
        "subjects": 9,
        "seeds": seeds,
        "shot_per_class": shot,
        "layer0_smoke": {"passed": smoke["passed"], "raw_shape": [22, 800], "cbramod_input_shape": [22, 4, 200], "feature_shape": [200]},
        "layer1": {
            "passed": l1.get("passed"),
            "bad_artifact_delta_acc_vs_clean": l1.get("BadArtifact_delta_acc_vs_clean"),
            "bad_content_delta_acc_vs_clean": l1.get("BadContent_delta_acc_vs_clean"),
            "bad_physio_delta_acc_vs_clean": l1.get("BadPhysio_delta_acc_vs_clean"),
            "bad_style_delta_acc_vs_clean": l1.get("BadStyle_delta_acc_vs_clean"),
        },
        "layer2": {
            "overall_auc": l2.get("overall_auc"),
            "bad_artifact_auc": l2.get("bad_artifact_auc"),
            "bad_content_auc": l2.get("bad_content_auc"),
            "bad_physio_auc": l2.get("bad_physio_auc"),
            "bad_style_auc": l2.get("bad_style_auc"),
            "top30_bad_rate": l2.get("top30_bad_rate"),
            "bottom30_bad_rate": l2.get("bottom30_bad_rate"),
            "passed": l2.get("passed"),
        },
        "layer3": {
            "sas_top50_minus_random50_acc": l3.get("sas_top50_minus_random50_acc"),
            "sas_top50_minus_naive_acc": l3.get("sas_top50_minus_naive_acc"),
            "sas_top50_minus_naive_ece": l3.get("sas_top50_minus_naive_ece"),
            "worst_subject_gain": None,
            "subject_std_change": None,
            "passed": l3.get("passed"),
        },
        "interpretation_level": interpretation,
        "decision": decision,
        "next_action": next_action,
        "protocol_leakage_detected": protocol_audit["protocol_leakage_detected"],
    }


def write_report(path, path_audit, smoke, l1_rows, l2, l3, paired, compact, old_audit):
    lines = [
        "# SAS-Cert-CBraMod MVE Report",
        "",
        "## Path Audit",
        "```json",
        json.dumps(path_audit, indent=2, sort_keys=True),
        "```",
        "",
        "## Smoke",
        "```json",
        json.dumps(smoke, indent=2, sort_keys=True)[:6000],
        "```",
        "",
        "## Layer 1",
        "```json",
        json.dumps(summarize_layer1(l1_rows), indent=2, sort_keys=True),
        "```",
        "",
        "## Layer 2",
        "```json",
        json.dumps(l2, indent=2, sort_keys=True),
        "```",
        "",
        "## Layer 3",
        "```json",
        json.dumps({"summary": l3, "paired": paired}, indent=2, sort_keys=True),
        "```",
        "",
        "## Old EEGNet Baseline Readonly Audit",
        "```json",
        json.dumps(old_audit, indent=2, sort_keys=True),
        "```",
        "",
        "## Compact Result",
        "```json",
        json.dumps(compact, indent=2, sort_keys=True),
        "```",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

