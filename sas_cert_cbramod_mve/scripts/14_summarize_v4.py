from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize v4 confirmatory MVE and physio forensics.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)
    outputs = project_root / "outputs"
    layer3 = read_json(outputs / "layer3_v4" / "layer3_v4_summary.json")
    physio = read_json(outputs / "physio_v4" / "physio_v4_summary.json")
    protocol = read_json(outputs / "audit" / "protocol_audit.json")
    v3 = read_json(outputs / "compact_result_v3.json")

    main_row = next(row for row in layer3["paired_comparison"] if row["group"] == "SASCert_SoftAR_LS010")
    artifact_row = next(row for row in layer3["paired_comparison"] if row["group"] == "ArtifactReject_raw_diagnostic")
    compact = {
        "project": "sas_cert_cbramod_mve",
        "stage": "v4_confirmatory_mve",
        "status": "completed",
        "backbone": "CBraMod",
        "backbone_frozen": True,
        "dataset": "BCIC-IV-2a",
        "main_method": "SASCert_SoftAR_LS010",
        "score_variant": "artifact_gate_content_rank",
        "artifact_reject_percentile": 10,
        "w_min": 0.2,
        "label_smoothing": 0.10,
        "baseline": "NaiveAug",
        "delta_acc": main_row["delta_acc"],
        "delta_macro_f1": main_row["delta_macro_f1"],
        "delta_ece": main_row["delta_ece"],
        "delta_nll": main_row["delta_nll"],
        "delta_brier": main_row["delta_brier"],
        "delta_worst_subject_acc": main_row["delta_worst_subject_acc"],
        "delta_subject_wise_std": main_row["delta_subject_wise_std"],
        "subject_win_rate_acc": main_row["subject_win_rate_acc"],
        "subject_win_rate_macro_f1": main_row["subject_win_rate_macro_f1"],
        "physio_best_component": physio["physio_best_component"],
        "physio_best_component_auc": physio["physio_best_component_auc"],
        "physio_ready_for_training": physio["physio_ready_for_training"],
        "protocol_leakage_detected": bool(protocol.get("protocol_leakage_detected", False)) or bool(layer3.get("protocol_leakage_detected", False)) or bool(physio.get("protocol_leakage_detected", False)),
        "decision": main_row["final_group_decision"],
        "next_action": next_action(main_row["final_group_decision"], physio),
    }
    (outputs / "compact_result_v4.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    write_report(outputs / "SAS_CERT_CBRAMOD_V4_CONFIRMATORY_REPORT.md", compact, layer3, physio, v3, artifact_row)
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def next_action(decision: str, physio: Dict[str, object]) -> str:
    if decision == "GO_STABLE_CALIBRATED_SOFTWEIGHT":
        action = "enter longer epoch or more seeds confirmatory run with SASCert_SoftAR_LS010"
    elif decision == "PARTIAL_GO_F1_STABLE":
        action = "keep method and prioritize Macro-F1 / balanced-performance reporting"
    elif decision == "CALIBRATION_REGRESSION":
        action = "return to label smoothing epsilon=0.05 or BrierMix"
    else:
        action = "do not expand; analyze score variant and support split stability"
    if not physio.get("physio_ready_for_training", False):
        action += "; keep physio out of training and refine subscore definitions"
    return action


def write_report(path: Path, compact: Dict[str, object], layer3: Dict[str, object], physio: Dict[str, object], v3: Dict[str, object], artifact_row: Dict[str, object]) -> None:
    lines = [
        "# SAS-Cert-CBraMod V4 Confirmatory MVE And Physio Forensics",
        "",
        "## Summary",
        "",
        f"- Status: `{compact['status']}`",
        f"- Main method: `{compact['main_method']}`",
        f"- Decision: `{compact['decision']}`",
        f"- Protocol leakage detected: `{compact['protocol_leakage_detected']}`",
        "",
        "## Why V3 Was GO",
        "",
        f"- V3 best group was `{v3['best_calibration_group']}`.",
        f"- V3 Macro-F1 delta was `{v3['best_group_delta_macro_f1']:+.4f}` with ECE delta `{v3['best_group_delta_ece']:+.4f}`.",
        f"- V3 NLL/Brier deltas were `{v3['best_group_delta_nll']:+.4f}` / `{v3['best_group_delta_brier']:+.4f}`.",
        "",
        "## V4 Confirmatory Result",
        "",
        "| Metric | Delta vs NaiveAug |",
        "| --- | ---: |",
        f"| Acc | {compact['delta_acc']:+.4f} |",
        f"| Macro-F1 | {compact['delta_macro_f1']:+.4f} |",
        f"| ECE | {compact['delta_ece']:+.4f} |",
        f"| NLL | {compact['delta_nll']:+.4f} |",
        f"| Brier | {compact['delta_brier']:+.4f} |",
        f"| Worst-subject Acc | {compact['delta_worst_subject_acc']:+.4f} |",
        f"| Subject-wise Std | {compact['delta_subject_wise_std']:+.4f} |",
        "",
        f"- Subject win rate Acc: `{compact['subject_win_rate_acc']:.4f}`",
        f"- Subject win rate Macro-F1: `{compact['subject_win_rate_macro_f1']:.4f}`",
        "",
        "## Artifact Diagnostic Branch",
        "",
        f"- ArtifactReject_raw_diagnostic Acc delta: `{artifact_row['delta_acc']:+.4f}`",
        f"- ArtifactReject_raw_diagnostic Macro-F1 delta: `{artifact_row['delta_macro_f1']:+.4f}`",
        f"- ArtifactReject_raw_diagnostic ECE delta: `{artifact_row['delta_ece']:+.4f}`",
        f"- Decision: `{artifact_row['final_group_decision']}`",
        "",
        "## Physio Forensics",
        "",
        "| Bad type | Subscore | AUC | Inverted AUC | Direction issue | Usable |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in physio["subscore_auc"]:
        lines.append(
            f"| {row['bad_type']} | {row['subscore']} | {row['original_auc']:.4f} | {row['inverted_auc']:.4f} | {row['direction_maybe_wrong']} | {row['usable_physio_component']} |"
        )
    lines.extend(
        [
            "",
            f"- Physio best component: `{compact['physio_best_component']}` AUC `{compact['physio_best_component_auc']:.4f}`",
            f"- Physio ready for training: `{compact['physio_ready_for_training']}`",
            "",
            "## Final Decision",
            "",
            f"`{compact['decision']}`",
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
