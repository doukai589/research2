from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


TEMPLATE_45 = [
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
]
BNCI22 = ["FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare/audit MIRepNet adapter without fabricating exact reproduction.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out = workspace / "outputs" / "paper_reproduction"
    proc = out / "mirepnet_processed"
    out.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    bcic_root = workspace.parent / "CBraMod-main" / "tmp_in" / "BCIC2a" / "MNE-bnci-data" / "database" / "data-sets" / "001-2014"
    mirepnet_root = workspace / "third_party" / "backbones" / "MIRepNet"
    missing = missing_modules(["wandb", "einops"])
    raw = audit_bcic_raw(bcic_root)
    present_template = [ch for ch in TEMPLATE_45 if ch in BNCI22]
    missing_template = [ch for ch in TEMPLATE_45 if ch not in BNCI22]
    manifest = {
        "adapter_ready": False,
        "reason": "not_generated_until_validated_inverse_distance_interpolation_and_EA_are_implemented",
        "local_backbone_path": str(mirepnet_root),
        "bcic_raw_path": str(bcic_root),
        "missing_dependencies": missing,
        "official_code_status": "README command only documents BNCI2014004; BNCI2014001 paper targets require unreleased/adapter pipeline.",
        "input_raw_shape": raw,
        "processed_shape_expected": {
            "BNCI2014001_2class_left_right": "[samples,45,1000]",
            "BNCI2014001_4class_full": "[samples,45,1000]",
        },
        "selected_tasks": ["BNCI2014001_2class_left_right", "BNCI2014001_4class_full"],
        "subject_count": 9,
        "class_count": {"BNCI2014001_2class_left_right": 2, "BNCI2014001_4class_full": 4},
        "sample_count_expected_from_raw": {"all_4class_trials": raw.get("valid_trials"), "left_right_subset_estimate": raw.get("left_right_trials")},
        "channel_template": TEMPLATE_45,
        "source_channels": BNCI22,
        "directly_observed_template_channels": present_template,
        "interpolated_template_channels_required": missing_template,
        "channel_template_mapping": "requires inverse-distance interpolation from 22 BNCI channels to 45-channel MIRepNet template",
        "preprocessing_required": ["8-30Hz bandpass", "resample to 250Hz", "4s window = 1000 points", "Euclidean Alignment"],
        "EA_behavior": "PAPER_PROTOCOL_EA requested; official behavior for train/test separation must be verified before generating exact arrays",
        "generated_files": [],
        "expected_files_not_generated": ["X.npy", "labels.npy", "metadata.json", "subject_split.json", "protocol_manifest.json"],
        "compatibility_with_official_code": "not_ready; official code expects ./data/<dataset>/X.npy and labels.npy plus dependencies",
        "protocol_classification": {
            "code_exact": "blocked_for_BNCI2014001_and_BNCI2014001-4",
            "paper_exact": "blocked_until_adapter_validated",
            "executable_hybrid": "not_created_to_avoid_mislabeling",
        },
    }
    write_json(out / "mirepnet_adapter_manifest.json", manifest)
    write_report(out / "mirepnet_adapter_audit.md", manifest)
    print(json.dumps({"status": "partial", "adapter_ready": False, "missing_dependencies": missing}, indent=2))


def audit_bcic_raw(root: Path) -> Dict[str, object]:
    try:
        import scipy.io
    except Exception as exc:
        return {"status": "missing_scipy", "error": str(exc)}
    files = sorted(root.glob("A0[1-9][TE].mat"))
    labels = []
    raw_shapes = []
    for path in files:
        mat = scipy.io.loadmat(path)
        runs = mat["data"][0]
        for run_idx in range(3, len(runs)):
            cell = runs[run_idx][0, 0]
            raw_shapes.append(list(cell[0].shape))
            y = (cell[2].reshape(-1).astype(int) - 1).tolist()
            labels.extend([int(v) for v in y if 0 <= int(v) <= 3])
    return {
        "status": "audited",
        "files": len(files),
        "raw_shape_examples": raw_shapes[:5],
        "valid_trials": len(labels),
        "left_right_trials": sum(1 for y in labels if y in {0, 1}),
        "labels": sorted(set(labels)),
    }


def missing_modules(names: List[str]) -> List[str]:
    out = []
    for name in names:
        try:
            __import__(name)
        except Exception:
            out.append(name)
    return out


def write_report(path: Path, manifest: Dict[str, object]) -> None:
    lines = [
        "# MIRepNet Adapter Audit",
        "",
        f"- adapter_ready: `{manifest['adapter_ready']}`",
        f"- reason: {manifest['reason']}",
        f"- missing_dependencies: `{manifest['missing_dependencies']}`",
        f"- raw BCIC path: `{manifest['bcic_raw_path']}`",
        f"- input_raw_shape: `{manifest['input_raw_shape']}`",
        f"- expected processed shape: `{manifest['processed_shape_expected']}`",
        f"- selected tasks: `{manifest['selected_tasks']}`",
        f"- directly observed template channels: `{manifest['directly_observed_template_channels']}`",
        f"- interpolated channels required: `{manifest['interpolated_template_channels_required']}`",
        f"- EA behavior: `{manifest['EA_behavior']}`",
        f"- generated files: `{manifest['generated_files']}`",
        "",
        "No X.npy/labels.npy were generated in this gate because doing so without a validated inverse-distance interpolation and exact EA split behavior would create a hybrid artifact that could be mistaken for paper_exact.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
