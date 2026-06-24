from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract paper targets and protocol registry for reproduction gate.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out = workspace / "outputs" / "paper_reproduction"
    out.mkdir(parents=True, exist_ok=True)

    targets = paper_targets()
    protocols = protocol_registry(workspace)
    conflicts = paper_code_conflicts(workspace)

    write_csv(out / "paper_targets.csv", targets)
    write_csv(out / "protocol_registry.csv", protocols)
    write_csv(out / "paper_code_conflict_table.csv", conflicts)
    write_json(out / "paper_targets.json", {"targets": targets, "protocols": protocols, "conflicts": conflicts})
    print(json.dumps({"status": "completed", "targets": len(targets), "protocols": len(protocols), "conflicts": len(conflicts)}, indent=2))


def paper_targets() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    def add(model: str, dataset: str, task: str, protocol: str, metric: str, mean: float, std: float, notes: str) -> None:
        rows.append({
            "model": model,
            "dataset": dataset,
            "task": task,
            "paper_protocol": protocol,
            "metric": metric,
            "paper_mean": mean,
            "paper_std": std,
            "notes": notes,
        })

    add("CBraMod", "PhysioNet-MI", "4-class", "paper_exact", "balanced_accuracy", 0.6417, 0.0091, "subjects 1-70 train, 71-89 val, 90-109 test; 4s; 200Hz")
    add("CBraMod", "PhysioNet-MI", "4-class", "paper_exact", "cohen_kappa", 0.5222, 0.0169, "subjects 1-70 train, 71-89 val, 90-109 test; 4s; 200Hz")
    add("CBraMod", "PhysioNet-MI", "4-class", "paper_exact", "weighted_f1", 0.6427, 0.0100, "subjects 1-70 train, 71-89 val, 90-109 test; 4s; 200Hz")
    add("CBraMod", "BCIC-IV-2a", "E.9 4-class", "paper_exact", "balanced_accuracy", 0.5138, 0.0066, "[2,6]s; 0.3-40Hz; 200Hz; subjects 1-5 train, 6-7 val, 8-9 test")
    add("CBraMod", "BCIC-IV-2a", "E.9 4-class", "paper_exact", "cohen_kappa", 0.3518, 0.0094, "[2,6]s; 0.3-40Hz; 200Hz; subjects 1-5 train, 6-7 val, 8-9 test")
    add("CBraMod", "BCIC-IV-2a", "E.9 4-class", "paper_exact", "weighted_f1", 0.4984, 0.0085, "[2,6]s; 0.3-40Hz; 200Hz; subjects 1-5 train, 6-7 val, 8-9 test")
    add("CBraMod", "BCIC-IV-2a LOSO key ingredients", "LOSO key ingredients", "paper_exact", "balanced_accuracy", 0.7405, 0.0635, "LOSO with EA, session statistics, Mixup, subject-wise regularization")
    add("CBraMod", "BCIC-IV-2a LOSO key ingredients", "LOSO key ingredients", "paper_exact", "auc_pr", 0.5997, 0.0833, "LOSO with EA, session statistics, Mixup, subject-wise regularization")
    add("CBraMod", "BCIC-IV-2a LOSO key ingredients", "LOSO key ingredients", "paper_exact", "auroc", 0.7195, 0.0682, "LOSO with EA, session statistics, Mixup, subject-wise regularization")
    add("MIRepNet", "BNCI2014001", "likely 2-class left/right", "paper_exact", "accuracy_full_finetune", 0.8177, 0.0027, "30% target-session trials; 8-30Hz; 250Hz; 45-channel template; EA")
    add("MIRepNet", "BNCI2014001", "likely 2-class left/right", "paper_exact", "accuracy_linear_probe", 0.7536, 0.0226, "30% target-session trials; 8-30Hz; 250Hz; 45-channel template; EA")
    add("MIRepNet", "BNCI2014001-4", "4-class full", "paper_exact", "accuracy_full_finetune", 0.6414, 0.0031, "30% target-session trials; 8-30Hz; 250Hz; 45-channel template; EA")
    add("MIRepNet", "BNCI2014001-4", "4-class full", "paper_exact", "accuracy_linear_probe", 0.5194, 0.0184, "30% target-session trials; 8-30Hz; 250Hz; 45-channel template; EA")
    return rows


def protocol_registry(workspace: Path) -> List[Dict[str, object]]:
    cbramod = workspace / "third_party" / "CBraMod-main"
    mirepnet = workspace / "third_party" / "backbones" / "MIRepNet"
    return [
        {"model": "CBraMod", "dataset": "PhysioNet-MI", "protocol_name": "code_exact", "protocol_type": "code_exact", "entrypoint": str(cbramod / "finetune_main.py"), "executable_now": True, "status": "available_but_requires_processed_lmdb", "notes": "Official entry defaults are code behavior; requires LMDB datasets_dir with train/val/test keys."},
        {"model": "CBraMod", "dataset": "PhysioNet-MI", "protocol_name": "paper_exact", "protocol_type": "paper_exact", "entrypoint": "paper raw EDF preprocessing + CBraMod full finetune", "executable_now": False, "status": "needs_paper_preprocessing_lmdb", "notes": "Raw EDF split 1-70/71-89/90-109 must be converted to exact LMDB before exact run."},
        {"model": "CBraMod", "dataset": "PhysioNet-MI", "protocol_name": "executable_hybrid", "protocol_type": "executable_hybrid", "entrypoint": "Step1 mapped/cropped forward smoke", "executable_now": True, "status": "forward_only_not_reproduction", "notes": "Can forward with 22-channel mapping, but this is not paper exact because paper uses 64 channels."},
        {"model": "CBraMod", "dataset": "BCIC-IV-2a", "protocol_name": "code_exact", "protocol_type": "code_exact", "entrypoint": str(cbramod / "finetune_main.py"), "executable_now": True, "status": "available_but_requires_processed_lmdb", "notes": "Official code uses LMDB dataset class; default CLI is not BCIC."},
        {"model": "CBraMod", "dataset": "BCIC-IV-2a", "protocol_name": "paper_exact", "protocol_type": "paper_exact", "entrypoint": "paper raw MAT preprocessing + CBraMod full finetune", "executable_now": False, "status": "needs_paper_preprocessing_lmdb", "notes": "Need exact [2,6]s, 0.3-40Hz, 200Hz, 1-5/6-7/8-9 split before exact run."},
        {"model": "CBraMod", "dataset": "BCIC-IV-2a", "protocol_name": "executable_hybrid", "protocol_type": "executable_hybrid", "entrypoint": "Step1 raw MAT loader + CBraMod forward smoke", "executable_now": True, "status": "forward_only_not_reproduction", "notes": "Forward works, but not full fine-tuning reproduction."},
        {"model": "MIRepNet", "dataset": "BNCI2014001", "protocol_name": "code_exact", "protocol_type": "code_exact", "entrypoint": str(mirepnet / "finetune.py"), "executable_now": False, "status": "code_not_released_for_dataset", "notes": "README command supports BNCI2014004; README says code for other datasets will be released."},
        {"model": "MIRepNet", "dataset": "BNCI2014001", "protocol_name": "paper_exact", "protocol_type": "paper_exact", "entrypoint": "paper 45-channel template + EA + 30% target fine-tune", "executable_now": False, "status": "needs_adapter_and_dependencies", "notes": "Requires adapter to X.npy/labels.npy and local dependencies wandb/einops."},
        {"model": "MIRepNet", "dataset": "BNCI2014001", "protocol_name": "executable_hybrid", "protocol_type": "executable_hybrid", "entrypoint": "local adapter audit only", "executable_now": False, "status": "not_ready", "notes": "Hybrid cannot be scored until adapter is validated."},
        {"model": "MIRepNet", "dataset": "BNCI2014001-4", "protocol_name": "code_exact", "protocol_type": "code_exact", "entrypoint": str(mirepnet / "finetune.py"), "executable_now": False, "status": "code_not_released_for_dataset", "notes": "Repository dataset.py has BNCI2014001-4 branch but README says other dataset code to be released; dependency/preprocessed files missing."},
        {"model": "MIRepNet", "dataset": "BNCI2014001-4", "protocol_name": "paper_exact", "protocol_type": "paper_exact", "entrypoint": "paper 45-channel template + EA + 30% target fine-tune", "executable_now": False, "status": "needs_adapter_and_dependencies", "notes": "Requires validated inverse-distance channel interpolation and EA behavior."},
        {"model": "MIRepNet", "dataset": "BNCI2014001-4", "protocol_name": "executable_hybrid", "protocol_type": "executable_hybrid", "entrypoint": "local adapter audit only", "executable_now": False, "status": "not_ready", "notes": "Hybrid cannot be scored until adapter is validated."},
    ]


def paper_code_conflicts(workspace: Path) -> List[Dict[str, object]]:
    return [
        {"model": "CBraMod", "dataset": "PhysioNet-MI", "paper_claim": "Paper protocol uses raw PhysioNet-MI 64 channels, 200Hz, 4s, subject 1-70/71-89/90-109 split.", "code_behavior": "Official finetune_main.py consumes processed LMDB via datasets/physio_dataset.py; default CLI points to MentalArithmetic path.", "conflict_type": "preprocessing_and_entrypoint", "chosen_protocol": "paper_exact blocked until exact LMDB is generated; code_exact kept separate", "reason": "LMDB contents are opaque unless regenerated/audited.", "expected_impact": "Large; split/preprocessing mismatch can dominate metrics."},
        {"model": "CBraMod", "dataset": "BCIC-IV-2a", "paper_claim": "Paper E.9 uses [2,6]s, 0.3-40Hz, 200Hz, subjects 1-5 train, 6-7 val, 8-9 test.", "code_behavior": "Official dataset class consumes LMDB; repository default CLI is not BCIC-IV-2a.", "conflict_type": "split_and_preprocessing", "chosen_protocol": "paper_exact blocked until exact LMDB is generated; hybrid forward smoke not scored", "reason": "Existing SAS-Cert loader uses different bandpass in prior MVE and frozen pooling.", "expected_impact": "Large; exact paper comparison invalid without exact preprocessing."},
        {"model": "CBraMod", "dataset": "BCIC-IV-2a LOSO", "paper_claim": "LOSO result requires EA, session statistics, Mixup, and subject-wise regularization.", "code_behavior": "No matching key-ingredients implementation found in the clean official entrypoint.", "conflict_type": "missing_method_components", "chosen_protocol": "not_run_missing_key_ingredients_implementation", "reason": "Running plain LOSO would not reproduce the paper table.", "expected_impact": "Large; paper target is much higher and depends on ingredients."},
        {"model": "MIRepNet", "dataset": "BNCI2014001", "paper_claim": "Paper reports BNCI2014001 2-class with 8-30Hz, 250Hz, 45-channel template, EA, 30% target fine-tune.", "code_behavior": "README command only shows BNCI2014004; README states code for other datasets will be released.", "conflict_type": "dataset_code_unreleased", "chosen_protocol": "paper_exact blocked; code_exact unavailable", "reason": "Cannot call official default for BNCI2014001 without inventing missing data pipeline.", "expected_impact": "Blocking; any local adapter is hybrid/paper reconstruction, not code exact."},
        {"model": "MIRepNet", "dataset": "BNCI2014001-4", "paper_claim": "Paper reports BNCI2014001 full 4-class with 45-channel template and EA.", "code_behavior": "dataset.py has a BNCI2014001-4 branch but expects ./data/BNCI2014001/X.npy and labels.npy; local files absent.", "conflict_type": "missing_processed_data_and_adapter", "chosen_protocol": "adapter audit before reproduction", "reason": "Direct 22-channel BCIC input would violate paper and code expectations.", "expected_impact": "Blocking until adapter is validated."},
        {"model": "MIRepNet", "dataset": "BNCI2014001/BNCI2014001-4", "paper_claim": "Paper preprocessing includes 45-channel template and EA.", "code_behavior": "Local environment lacks wandb/einops; model PatchEmbedding defaults to num_channels=45.", "conflict_type": "dependency_and_input_shape", "chosen_protocol": "record missing dependency; no global install", "reason": "Attachment forbids global pollution; adapter must preserve 45 channels.", "expected_impact": "Blocking for code execution; high risk if bypassed."},
    ]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
