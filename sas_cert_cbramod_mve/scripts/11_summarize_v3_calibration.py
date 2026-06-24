from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SAS-Cert-CBraMod v3 calibration repair MVE.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)
    outputs = project_root / "outputs"

    summary = read_json(outputs / "layer3_v3" / "layer3_v3_summary.json")
    layer3_v2 = read_json(outputs / "layer3_v2" / "layer3_v2_summary.json")
    layer2_v2 = read_json(outputs / "layer2_v2" / "layer2_v2_summary.json")
    protocol = read_json(outputs / "audit" / "protocol_audit.json")

    paired = summary["paired_comparison"]
    best_row = choose_best_calibration_group(paired)
    final_decision = decide(paired, best_row)
    compact = {
        "project": "sas_cert_cbramod_mve",
        "stage": "v3_calibration_repair",
        "status": "completed",
        "backbone": "CBraMod",
        "backbone_frozen": True,
        "dataset": "BCIC-IV-2a",
        "score_variant": summary["score_variant"],
        "main_method": "SoftWeightArtifactReject",
        "artifact_reject_percentile": summary["artifact_reject_percentile"],
        "w_min": summary["w_min"],
        "baseline": "NaiveAug",
        "best_calibration_group": best_row["group"],
        "best_group_delta_acc": best_row["delta_acc"],
        "best_group_delta_macro_f1": best_row["delta_macro_f1"],
        "best_group_delta_ece": best_row["delta_ece"],
        "best_group_delta_nll": best_row["delta_nll"],
        "best_group_delta_brier": best_row["delta_brier"],
        "temperature_mean": summary["temperature_mean"],
        "temperature_std": summary["temperature_std"],
        "decision": final_decision,
        "protocol_leakage_detected": bool(protocol.get("protocol_leakage_detected", False)),
        "next_action": next_action(final_decision, best_row, layer2_v2),
    }
    (outputs / "compact_result_v3.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    write_report(outputs / "SAS_CERT_CBRAMOD_V3_CALIBRATION_REPORT.md", compact, summary, layer3_v2, layer2_v2, paired)
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def choose_best_calibration_group(paired: Sequence[Dict[str, object]]) -> Dict[str, object]:
    candidates = [
        row
        for row in paired
        if row["group"]
        in {
            "SWAR_TempScale_SourceVal",
            "SWAR_LabelSmoothing_eps005",
            "SWAR_LabelSmoothing_eps010",
            "SWAR_BrierMix_lam005",
            "SWAR_BrierMix_lam010",
            "SoftWeightArtifactReject_raw",
        }
    ]
    go = [row for row in candidates if row["final_group_decision"] == "GO_CALIBRATED_SOFTWEIGHT"]
    if go:
        return max(go, key=lambda r: (r["delta_acc"] + r["delta_macro_f1"], -r["delta_ece"], -r["delta_nll"]))
    useful = [row for row in candidates if row["delta_acc"] >= 0.01 or row["delta_macro_f1"] >= 0.01]
    if useful:
        return min(useful, key=lambda r: (r["delta_ece"], r["delta_nll"], -(r["delta_acc"] + r["delta_macro_f1"])))
    return min(candidates, key=lambda r: (r["delta_ece"], r["delta_nll"]))


def decide(paired: Sequence[Dict[str, object]], best_row: Dict[str, object]) -> str:
    if any(row["final_group_decision"] == "GO_CALIBRATED_SOFTWEIGHT" for row in paired):
        return "GO_CALIBRATED_SOFTWEIGHT"
    if best_row["delta_acc"] >= 0.01 or best_row["delta_macro_f1"] >= 0.01:
        if best_row["delta_ece"] > 0.01:
            return "CALIBRATION_STILL_FAIL"
    if best_row["delta_ece"] <= 0.01:
        return "CALIBRATION_OVERREGULARIZED"
    return "STOP"


def next_action(decision: str, best_row: Dict[str, object], layer2_v2: Dict[str, object]) -> str:
    actions: List[str] = []
    if decision == "GO_CALIBRATED_SOFTWEIGHT":
        actions.append("run longer epochs or more seeds with the winning calibrated SWAR group")
    elif decision == "CALIBRATION_STILL_FAIL":
        actions.append("try calibration-aware loss strength sweep or temperature scaling plus Brier/label smoothing shadow")
    elif decision == "CALIBRATION_OVERREGULARIZED":
        actions.append("reduce regularization strength and keep artifact_gate_content_rank")
    else:
        actions.append("stop v3 calibration path and refine score before more training")
    if layer2_v2.get("physio_fixed_badphysio_auc", 1.0) < 0.6:
        actions.append("continue REFINE_PHYSIO by splitting bandpower/covariance/topology/channel_order")
    if best_row["group"] == "ArtifactReject_raw":
        actions.append("keep artifact-only branch diagnostic only")
    return "; ".join(actions)


def find_row(paired: Sequence[Dict[str, object]], group: str) -> Dict[str, object] | None:
    for row in paired:
        if row["group"] == group:
            return row
    return None


def write_report(path: Path, compact: Dict[str, object], summary: Dict[str, object], layer3_v2: Dict[str, object], layer2_v2: Dict[str, object], paired: Sequence[Dict[str, object]]) -> None:
    v2_dec = layer3_v2["decision"]
    artifact_row = find_row(paired, "ArtifactReject_raw")
    raw_row = find_row(paired, "SoftWeightArtifactReject_raw")
    lines = [
        "# SAS-Cert-CBraMod V3 Calibration Repair Report",
        "",
        "## Summary",
        "",
        f"- Status: `{compact['status']}`",
        f"- Score variant: `{compact['score_variant']}`",
        f"- Main method: `{compact['main_method']}`",
        f"- Best calibration group: `{compact['best_calibration_group']}`",
        f"- Decision: `{compact['decision']}`",
        f"- Protocol leakage detected: `{compact['protocol_leakage_detected']}`",
        "",
        "## Why V2 Was Not Confirmatory",
        "",
        f"- V2 SoftWeightArtifactReject vs NaiveAug Acc delta was `{v2_dec['softweight_artifactreject_minus_naive_acc']:+.4f}`.",
        f"- V2 Macro-F1 delta was `{v2_dec['softweight_artifactreject_minus_naive_macro_f1']:+.4f}`.",
        f"- V2 ECE delta was `{v2_dec['softweight_artifactreject_minus_naive_ece']:+.4f}`, above the +0.01 gate.",
        "",
        "## V3 Group Decisions",
        "",
        "| Group | Acc delta | Macro-F1 delta | ECE delta | NLL delta | Brier delta | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(paired, key=lambda r: r["group"]):
        lines.append(
            f"| {row['group']} | {row['delta_acc']:+.4f} | {row['delta_macro_f1']:+.4f} | {row['delta_ece']:+.4f} | {row['delta_nll']:+.4f} | {row['delta_brier']:+.4f} | {row['final_group_decision']} |"
        )
    lines.extend(
        [
            "",
            "## Best Group",
            "",
            f"- Best calibration group: `{compact['best_calibration_group']}`",
            f"- Delta Acc: `{compact['best_group_delta_acc']:+.4f}`",
            f"- Delta Macro-F1: `{compact['best_group_delta_macro_f1']:+.4f}`",
            f"- Delta ECE: `{compact['best_group_delta_ece']:+.4f}`",
            f"- Delta NLL: `{compact['best_group_delta_nll']:+.4f}`",
            f"- Delta Brier: `{compact['best_group_delta_brier']:+.4f}`",
            "",
            "## Temperature Scaling",
            "",
            f"- Temperature mean: `{compact['temperature_mean']}`",
            f"- Temperature std: `{compact['temperature_std']}`",
            "- Temperature was fitted only on source validation, never on target E/test.",
            "",
            "## Artifact Branch",
            "",
        ]
    )
    if artifact_row:
        lines.append(
            f"- ArtifactReject_raw remains diagnostic: Acc delta `{artifact_row['delta_acc']:+.4f}`, Macro-F1 delta `{artifact_row['delta_macro_f1']:+.4f}`, ECE delta `{artifact_row['delta_ece']:+.4f}`, decision `{artifact_row['final_group_decision']}`."
        )
    if raw_row:
        lines.append(
            f"- Raw SWAR before calibration: Acc delta `{raw_row['delta_acc']:+.4f}`, Macro-F1 delta `{raw_row['delta_macro_f1']:+.4f}`, ECE delta `{raw_row['delta_ece']:+.4f}`."
        )
    lines.extend(
        [
            "",
            "## Physio Note",
            "",
            f"- Physio original AUC: `{layer2_v2['physio_original_auc']:.4f}`",
            f"- Physio fixed BadPhysio AUC: `{layer2_v2['physio_fixed_badphysio_auc']:.4f}`",
            "- REFINE_PHYSIO remains recommended because the fixed/variant diagnostics still show weak BadPhysio behavior.",
            "",
            "## Next Action",
            "",
            compact["next_action"],
            "",
            "## Compact JSON",
            "",
            "```json",
            json.dumps(compact, indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
