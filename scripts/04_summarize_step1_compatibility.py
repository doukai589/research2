from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Step1 compatibility smoke audit.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out_dir = workspace / "outputs" / "setup_audit_step1"
    rows = read_csv(out_dir / "backbone_dataset_compatibility.csv")
    bcic = read_json(out_dir / "bcic2a_loader_smoke.json")
    physio = read_json(out_dir / "physionetmi_loader_smoke.json")
    compact = build_compact(workspace, rows, bcic, physio)
    write_json(out_dir / "compact_step1_result.json", compact)
    write_report(out_dir / "STEP1_COMPATIBILITY_REPORT.md", rows, bcic, physio, compact)
    print(json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def build_compact(workspace: Path, rows: Sequence[Dict[str, str]], bcic: Dict[str, object], physio: Dict[str, object]) -> Dict[str, object]:
    ready_pairs = [pair_name(r) for r in rows if r["status"] == "ready"]
    needs_adapter = [pair_name(r) for r in rows if r["status"] in {"needs_adapter", "code_ready_weight_ready"}]
    failed = [pair_name(r) for r in rows if r["status"] == "failed"]
    warnings = []
    for r in rows:
        if r.get("missing_dependencies"):
            warnings.append(f"{pair_name(r)} missing_dependencies={r['missing_dependencies']}")
        if r["status"] in {"needs_adapter", "code_ready_weight_ready", "failed"}:
            warnings.append(f"{pair_name(r)} status={r['status']}: {r.get('notes', '')[:180]}")

    cb_bcic = find_pair(rows, "CBraMod", "BCIC-IV-2a")
    mi_bcic = find_pair(rows, "MIRepNet", "BCIC-IV-2a")
    cb_physio = find_pair(rows, "CBraMod", "PhysioNetMI")
    if mi_bcic and mi_bcic["status"] == "ready":
        rec = {
            "backbone": "MIRepNet",
            "dataset": "BCIC-IV-2a / BNCI2014-001",
            "reason": "MIRepNet x BCIC-IV-2a is ready; next should be frozen smoke-to-mini-eval, not full experiment.",
        }
    elif cb_bcic and cb_bcic["status"] == "ready" and (not mi_bcic or mi_bcic["status"] in {"needs_adapter", "failed"}):
        rec = {
            "backbone": "CBraMod",
            "dataset": "BCIC-IV-2a / BNCI2014-001",
            "reason": "CBraMod x BCIC-IV-2a is ready while MIRepNet needs adapter/failed; continue CBraMod + BCIC-IV-2a v5 locked confirmatory.",
        }
    elif cb_physio and cb_physio["status"] == "ready":
        rec = {
            "backbone": "CBraMod",
            "dataset": "PhysioNetMI / EEGMMI",
            "reason": "CBraMod x PhysioNetMI is ready; use it as second MI dataset after a 2-class MI loader mini-eval.",
        }
    else:
        rec = {
            "backbone": "",
            "dataset": "",
            "reason": "No ready priority pair; write adapter before training.",
        }

    status = "completed" if rows else "blocked"
    return {
        "stage": "step1_backbone_dataset_smoke",
        "status": status,
        "workspace_root": str(workspace),
        "tested_pairs": [pair_name(r) for r in rows],
        "ready_pairs": ready_pairs,
        "needs_adapter_pairs": needs_adapter,
        "failed_pairs": failed,
        "recommended_next": rec,
        "bcic2a": {
            "status": bcic.get("status", ""),
            "path": bcic.get("path", ""),
            "subjects": len(bcic.get("subjects", []) or []),
            "sessions": bcic.get("sessions", []),
            "trial_shape": bcic.get("trial_shape", ""),
            "labels": bcic.get("labels", []),
            "ready": bcic.get("ready", None),
        },
        "physionetmi": {
            "status": physio.get("status", ""),
            "path": physio.get("path", ""),
            "subjects": physio.get("subjects", None),
            "runs_per_subject": physio.get("runs_per_subject", None),
            "mi_runs_detected": physio.get("mi_runs_detected", []),
            "recommended_task": physio.get("recommended_task", ""),
            "ready": physio.get("ready", None),
        },
        "protocol_safety": {
            "raw_dataset_copied": False,
            "training_started": False,
            "old_outputs_modified": False,
        },
        "warnings": warnings,
    }


def find_pair(rows: Sequence[Dict[str, str]], backbone: str, dataset: str) -> Dict[str, str] | None:
    for row in rows:
        if row["backbone"] == backbone and row["dataset"] == dataset:
            return row
    return None


def pair_name(row: Dict[str, str]) -> str:
    return f"{row['backbone']} x {row['dataset']}"


def write_report(path: Path, rows: Sequence[Dict[str, str]], bcic: Dict[str, object], physio: Dict[str, object], compact: Dict[str, object]) -> None:
    lines = [
        "# Step1 Backbone Dataset Compatibility",
        "",
        "## Compatibility Matrix",
        "",
        "| Backbone | Dataset | Status | Forward | Feature | Missing deps | Adapter flags | Recommended | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        flags = []
        if truthy(r.get("channel_mapping_needed")):
            flags.append("channel_mapping")
        if truthy(r.get("resampling_needed")):
            flags.append("resampling")
        if truthy(r.get("crop_needed")):
            flags.append("crop")
        lines.append(
            f"| {r['backbone']} | {r['dataset']} | {r['status']} | {r['forward_success']} | `{r['feature_shape']}` | "
            f"{r['missing_dependencies']} | {', '.join(flags) or '-'} | {r['recommended_for_next']} | {r['notes'][:220]} |"
        )
    lines.extend([
        "",
        "## BCIC-IV-2a Loader",
        "",
        f"- status: `{bcic.get('status')}`",
        f"- path: `{bcic.get('path')}`",
        f"- subjects: `{bcic.get('subjects')}`",
        f"- sessions: `{bcic.get('sessions')}`",
        f"- trial_shape: `{bcic.get('trial_shape')}`",
        f"- labels: `{bcic.get('labels')}`",
        f"- EOG excluded: `{bcic.get('eog_channels_excluded')}`",
        "",
        "## PhysioNetMI Loader",
        "",
        f"- status: `{physio.get('status')}`",
        f"- path: `{physio.get('path')}`",
        f"- subjects: `{physio.get('subjects')}`",
        f"- runs_per_subject: `{physio.get('runs_per_subject')}`",
        f"- mi_runs_detected: `{physio.get('mi_runs_detected')}`",
        f"- recommended_task: {physio.get('recommended_task')}",
        f"- recommended_window: {physio.get('recommended_window')}",
        "",
        "## Recommended Next",
        "",
        f"- backbone: `{compact['recommended_next']['backbone']}`",
        f"- dataset: `{compact['recommended_next']['dataset']}`",
        f"- reason: {compact['recommended_next']['reason']}",
        "",
        "## Protocol Safety",
        "",
        f"- raw_dataset_copied: `{compact['protocol_safety']['raw_dataset_copied']}`",
        f"- training_started: `{compact['protocol_safety']['training_started']}`",
        f"- old_outputs_modified: `{compact['protocol_safety']['old_outputs_modified']}`",
        "",
        "## Warnings",
        "",
    ])
    lines.extend([f"- {w}" for w in compact["warnings"]] or ["- none"])
    lines.extend(["", "## Compact JSON", "", "```json", json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n")


def truthy(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
