from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


WORKSPACE = Path.cwd().resolve()
OUT_DIR = WORKSPACE / "outputs" / "setup_audit"


def main() -> None:
    backbones = read_csv(OUT_DIR / "backbone_inventory.csv")
    datasets_payload = json.loads((OUT_DIR / "dataset_inventory.json").read_text())
    datasets = datasets_payload["datasets"]
    rec = datasets_payload["recommended_dataset_order"]
    compact = build_compact(backbones, datasets, rec)
    (OUT_DIR / "compact_setup_result.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    write_report(OUT_DIR / "SETUP_AUDIT_REPORT.md", backbones, datasets, rec, compact)
    print(json.dumps(compact, indent=2, sort_keys=True))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_compact(backbones: Sequence[Dict[str, str]], datasets: Sequence[Dict[str, object]], rec: Dict[str, object]) -> Dict[str, object]:
    bb = {row["model_name"]: {"status": row["status"], "local_path": row["local_path"], "has_weights": str(row["has_pretrained_weight"]).lower() == "true", "git_commit": row["git_commit"]} for row in backbones}
    for name in ["CBraMod", "MIRepNet", "LaBraM", "EEGPT", "MFrFM", "EEG-DINO"]:
        bb.setdefault(name, {"status": "unresolved", "local_path": "", "has_weights": None, "git_commit": ""})
    first = rec.get("first") or {}
    second = rec.get("second") or {}
    third = rec.get("third") or {}
    warnings = []
    for row in backbones:
        if row["status"] in {"weight_missing", "unresolved", "blocked"}:
            warnings.append(f"{row['model_name']}: {row['status']}")
    primary_backbone = "MIRepNet" if bb.get("MIRepNet", {}).get("status") == "ready" else "CBraMod"
    blocked_by = []
    if primary_backbone == "MIRepNet" and not bb["MIRepNet"]["has_weights"]:
        blocked_by.append("MIRepNet weight missing")
    return {
        "stage": "step0_backbone_dataset_inventory",
        "status": "completed",
        "workspace_root": str(WORKSPACE),
        "project_root": str(WORKSPACE),
        "backbones": bb,
        "datasets": {
            "primary_mi_dataset": first.get("inferred_dataset_name", "") if isinstance(first, dict) else "",
            "primary_mi_path": first.get("root_path", "") if isinstance(first, dict) else "",
            "secondary_mi_dataset": second.get("inferred_dataset_name", "") if isinstance(second, dict) else "",
            "secondary_mi_path": second.get("root_path", "") if isinstance(second, dict) else "",
            "third_mi_dataset": third.get("inferred_dataset_name", "") if isinstance(third, dict) else "",
            "third_mi_path": third.get("root_path", "") if isinstance(third, dict) else "",
            "non_mi_candidates": rec.get("non_mi_candidates", []),
        },
        "recommended_next_experiment": {
            "backbone": primary_backbone,
            "dataset": first.get("inferred_dataset_name", "") if isinstance(first, dict) else "",
            "reason": "Use MIRepNet first if official weight is ready; otherwise continue with CBraMod on the complete BCIC-IV-2a inventory.",
            "blocked_by": blocked_by,
        },
        "warnings": warnings,
        "protocol_safety": {
            "raw_dataset_copied": False,
            "old_outputs_modified": False,
            "training_run_started": False,
        },
    }


def write_report(path: Path, backbones: Sequence[Dict[str, str]], datasets: Sequence[Dict[str, object]], rec: Dict[str, object], compact: Dict[str, object]) -> None:
    lines = [
        "# Step0 Backbone And Dataset Inventory",
        "",
        "## Paths",
        "",
        f"- workspace_root: `{WORKSPACE}`",
        f"- project_root: `{WORKSPACE}`",
        "",
        "## Backbone Inventory",
        "",
        "| Model | Status | Has weights | Commit | Local path | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in backbones:
        lines.append(f"| {row['model_name']} | {row['status']} | {row['has_pretrained_weight']} | `{row['git_commit'][:12]}` | `{row['local_path']}` | {row['notes'][:180]} |")
    display_datasets = [row for row in datasets if row.get("priority") != "exclude"][:80]
    lines.extend(["", "## Dataset Inventory", "", "| Dataset | Task | Priority | Status | Files | Format | Path |", "| --- | --- | --- | --- | ---: | --- | --- |"])
    for row in display_datasets:
        lines.append(f"| {row['inferred_dataset_name']} | {row['task_type']} | {row['priority']} | {row['status']} | {row['file_count']} | {row['format']} | `{row['root_path']}` |")
    if len(datasets) > len(display_datasets):
        lines.append(f"| ... | ... | ... | ... | ... | ... | omitted {len(datasets) - len(display_datasets)} low-priority/excluded scan groups; see dataset_inventory.csv/json |")
    lines.extend(["", "## MI Dataset Order", ""])
    for label in ["first", "second", "third"]:
        row = rec.get(label)
        if isinstance(row, dict):
            lines.append(f"- {label}: **{row['inferred_dataset_name']}** at `{row['root_path']}` status `{row['status']}` score `{row['mi_priority_score']}`")
        else:
            lines.append(f"- {label}: none")
    lines.extend(
        [
            "",
            "## Recommended Next Experiment",
            "",
            f"- backbone: `{compact['recommended_next_experiment']['backbone']}`",
            f"- dataset: `{compact['recommended_next_experiment']['dataset']}`",
            f"- reason: {compact['recommended_next_experiment']['reason']}",
            f"- blocked_by: `{compact['recommended_next_experiment']['blocked_by']}`",
            "",
            "## Protocol Safety",
            "",
            f"- raw_dataset_copied: `{compact['protocol_safety']['raw_dataset_copied']}`",
            f"- old_outputs_modified: `{compact['protocol_safety']['old_outputs_modified']}`",
            f"- training_run_started: `{compact['protocol_safety']['training_run_started']}`",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {w}" for w in compact["warnings"]] or ["- none"])
    lines.extend(["", "## Compact JSON", "", "```json", json.dumps(compact, indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
