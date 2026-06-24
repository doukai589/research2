from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper reproduction gate.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out = workspace / "outputs" / "paper_reproduction"
    out.mkdir(parents=True, exist_ok=True)
    cbr = read_csv_optional(out / "cbramod_metrics.csv")
    mir = read_csv_optional(out / "mirepnet_metrics.csv")
    disc = read_csv_optional(out / "cbramod_discrepancy.csv") + read_csv_optional(out / "mirepnet_discrepancy.csv")
    conflicts = read_csv_optional(out / "paper_code_conflict_table.csv")
    targets = read_csv_optional(out / "paper_targets.csv")
    adapter = read_json_optional(out / "mirepnet_adapter_manifest.json")

    all_metrics = cbr + mir
    write_csv(out / "reproduction_metrics.csv", all_metrics)
    write_csv(out / "discrepancy_table.csv", normalize_discrepancies(disc))
    compact = build_compact(cbr, mir, targets, adapter)
    write_json(out / "compact_paper_repro_result.json", compact)
    write_report(out / "PAPER_REPRODUCTION_GATE_REPORT.md", cbr, mir, conflicts, compact)
    print(json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True))


def build_compact(cbr: List[Dict[str, str]], mir: List[Dict[str, str]], targets: List[Dict[str, str]], adapter: Dict[str, object]) -> Dict[str, object]:
    cbr_phys = task_status(cbr, "CBraMod", "PhysioNet-MI", "paper_exact")
    cbr_bcic = task_status(cbr, "CBraMod", "BCIC-IV-2a", "paper_exact")
    cbr_loso = task_status(cbr, "CBraMod", "BCIC-IV-2a LOSO key ingredients", "paper_exact")
    mir_2 = task_status(mir, "MIRepNet", "BNCI2014001", "paper_exact")
    mir_4 = task_status(mir, "MIRepNet", "BNCI2014001-4", "paper_exact")
    warnings = []
    if not cbr_phys["ran"] or not cbr_bcic["ran"]:
        warnings.append("CBraMod paper_exact did not run; only gate/failure audit is available.")
    if not adapter.get("adapter_ready"):
        warnings.append("MIRepNet adapter not ready; MIRepNet paper_exact did not run.")
    decision = "ADAPTER_NEEDED"
    safe = False
    next_action = "Build exact preprocessing/LMDB for CBraMod paper runs and validate MIRepNet 45-channel interpolation+EA adapter before claiming paper reproduction."
    if cbr_phys["ran"] and cbr_bcic["ran"] and adapter.get("adapter_ready"):
        decision = "DISCREPANCY_AUDIT_NEEDED"
    elif cbr_phys["ran"] and cbr_bcic["ran"]:
        decision = "CBRAMOD_OK_MIREPNET_PENDING"
        safe = True
        next_action = "CBraMod can continue as SAS-Cert backbone; MIRepNet remains pending."
    return {
        "stage": "paper_reproduction_gate",
        "status": "partial",
        "cbramod": {
            "physionet_mi": {
                "ran": cbr_phys["ran"],
                "matched_or_close": cbr_phys["matched_or_close"],
                "local_metrics": cbr_phys["local_metrics"],
                "paper_metrics": paper_metrics(targets, "CBraMod", "PhysioNet-MI"),
                "status": cbr_phys["status"],
            },
            "bcic_iv_2a_e9": {
                "ran": cbr_bcic["ran"],
                "matched_or_close": cbr_bcic["matched_or_close"],
                "local_metrics": cbr_bcic["local_metrics"],
                "paper_metrics": paper_metrics(targets, "CBraMod", "BCIC-IV-2a"),
                "status": cbr_bcic["status"],
            },
            "bcic_iv_2a_loso_key_ingredients": {
                "ran": cbr_loso["ran"],
                "reason": cbr_loso["status"] or "not_run_missing_key_ingredients_implementation",
            },
        },
        "mirepnet": {
            "adapter_ready": bool(adapter.get("adapter_ready")),
            "bnci2014001": {
                "ran": mir_2["ran"],
                "matched_or_close": mir_2["matched_or_close"],
                "local_metrics": mir_2["local_metrics"],
                "paper_metrics": paper_metrics(targets, "MIRepNet", "BNCI2014001"),
                "status": mir_2["status"],
            },
            "bnci2014001_4": {
                "ran": mir_4["ran"],
                "matched_or_close": mir_4["matched_or_close"],
                "local_metrics": mir_4["local_metrics"],
                "paper_metrics": paper_metrics(targets, "MIRepNet", "BNCI2014001-4"),
                "status": mir_4["status"],
            },
        },
        "overall_decision": decision,
        "safe_for_sas_cert_next_stage": safe,
        "warnings": warnings,
        "next_action": next_action,
    }


def task_status(rows: List[Dict[str, str]], model: str, dataset: str, protocol: str) -> Dict[str, object]:
    subset = [r for r in rows if r.get("model") == model and r.get("dataset") == dataset and r.get("protocol_name") == protocol]
    ran = bool(subset) and all(r.get("local_mean") not in {"", None} for r in subset)
    local = {r.get("metric", ""): to_float_or_string(r.get("local_mean", "")) for r in subset if r.get("metric") and r.get("local_mean")}
    statuses = sorted(set(r.get("status", "") for r in subset))
    return {
        "ran": ran,
        "matched_or_close": None if not ran else all(s in {"matched", "close"} for s in statuses),
        "local_metrics": local,
        "status": ";".join(statuses) if statuses else "not_run",
    }


def paper_metrics(targets: List[Dict[str, str]], model: str, dataset: str) -> Dict[str, object]:
    return {r["metric"]: {"mean": float(r["paper_mean"]), "std": float(r["paper_std"])} for r in targets if r["model"] == model and r["dataset"] == dataset}


def normalize_discrepancies(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    cols = ["model", "dataset", "protocol_name", "metric", "paper_mean", "paper_std", "local_mean", "local_std", "absolute_diff", "relative_diff", "within_2pp", "within_5pp", "status", "likely_reason"]
    return [{c: r.get(c, "") for c in cols} for r in rows]


def write_report(path: Path, cbr: List[Dict[str, str]], mir: List[Dict[str, str]], conflicts: List[Dict[str, str]], compact: Dict[str, object]) -> None:
    lines = [
        "# Paper Reproduction Gate Report",
        "",
        "This report keeps `code_exact`, `paper_exact`, and `executable_hybrid` separate. Hybrid outputs are not treated as exact reproduction.",
        "",
        "## CBraMod Results",
        "",
    ]
    append_protocol_sections(lines, cbr)
    lines.extend(["", "## MIRepNet Results", ""])
    append_protocol_sections(lines, mir)
    lines.extend(["", "## Paper-Code Conflict Table", "", "| Model | Dataset | Conflict | Chosen Protocol | Expected Impact |", "| --- | --- | --- | --- | --- |"])
    for r in conflicts:
        lines.append(f"| {r.get('model')} | {r.get('dataset')} | {r.get('conflict_type')} | {r.get('chosen_protocol')} | {r.get('expected_impact')} |")
    lines.extend([
        "",
        "## Discrepancy Analysis",
        "",
        "- CBraMod paper_exact: not run; likely reason is missing exact regenerated/audited LMDB preprocessing and full 50-epoch fine-tune.",
        "- MIRepNet paper_exact: not run; likely reason is adapter/dependency and unreleased official BNCI2014001 pipeline.",
        "- Executable hybrid: only forward/audit status exists; not a reproduction result.",
        "",
        "## Decision",
        "",
        f"- overall_decision: `{compact['overall_decision']}`",
        f"- safe_for_sas_cert_next_stage: `{compact['safe_for_sas_cert_next_stage']}`",
        f"- next_action: {compact['next_action']}",
        "",
        "## Compact JSON",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n")


def append_protocol_sections(lines: List[str], rows: List[Dict[str, str]]) -> None:
    for protocol in ["code_exact", "paper_exact", "executable_hybrid"]:
        lines.append(f"### {protocol}")
        lines.append("| Dataset | Task | Metric | Paper | Local | Status | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in [r for r in rows if r.get("protocol_name") == protocol]:
            paper = f"{r.get('paper_mean')}±{r.get('paper_std')}" if r.get("paper_mean") else ""
            local = f"{r.get('local_mean')}±{r.get('local_std')}" if r.get("local_mean") else ""
            lines.append(f"| {r.get('dataset')} | {r.get('task')} | {r.get('metric')} | {paper} | {local} | {r.get('status')} | {r.get('likely_reason')} |")
        lines.append("")


def to_float_or_string(value: str) -> object:
    try:
        return float(value)
    except Exception:
        return value


def read_csv_optional(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_optional(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
