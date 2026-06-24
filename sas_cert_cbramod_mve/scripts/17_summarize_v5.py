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
    parser = argparse.ArgumentParser(description="Summarize v5 locked-method confirmatory run.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)
    outputs = project_root / "outputs"
    layer3 = read_json(outputs / "layer3_v5" / "layer3_v5_summary.json")
    physio = read_json(outputs / "physio_v5" / "physio_v5_summary.json")
    protocol = read_json(outputs / "audit" / "protocol_audit.json")
    v4 = read_json(outputs / "compact_result_v4.json")
    raw = find_pair(layer3["paired_comparison"], "SASCert_SoftAR_LS010", "NaiveAug_raw")
    ls = find_pair(layer3["paired_comparison"], "SASCert_SoftAR_LS010", "NaiveAug_LS010")
    artifact_raw = find_pair(layer3["paired_comparison"], "ArtifactReject_raw_diagnostic", "NaiveAug_raw")
    decision = decide(raw, ls)
    compact = {
        "project": "sas_cert_cbramod_mve",
        "stage": "v5_locked_confirmatory",
        "status": "completed",
        "backbone": "CBraMod",
        "backbone_frozen": True,
        "dataset": "BCIC-IV-2a",
        "main_method": "SASCert_SoftAR_LS010",
        "score_variant": "artifact_gate_content_rank",
        "artifact_reject_percentile": 10,
        "w_min": 0.2,
        "label_smoothing": 0.10,
        "baseline_raw": "NaiveAug_raw",
        "baseline_ls": "NaiveAug_LS010",
        "delta_acc_vs_raw": raw["delta_acc"],
        "delta_macro_f1_vs_raw": raw["delta_macro_f1"],
        "delta_ece_vs_raw": raw["delta_ece"],
        "delta_nll_vs_raw": raw["delta_nll"],
        "delta_brier_vs_raw": raw["delta_brier"],
        "delta_acc_vs_ls": ls["delta_acc"],
        "delta_macro_f1_vs_ls": ls["delta_macro_f1"],
        "delta_ece_vs_ls": ls["delta_ece"],
        "delta_nll_vs_ls": ls["delta_nll"],
        "delta_brier_vs_ls": ls["delta_brier"],
        "subject_win_rate_acc_vs_raw": raw["subject_win_rate_acc"],
        "subject_win_rate_macro_f1_vs_raw": raw["subject_win_rate_macro_f1"],
        "seed_win_rate_acc_vs_raw": raw["seed_win_rate_acc"],
        "seed_win_rate_macro_f1_vs_raw": raw["seed_win_rate_macro_f1"],
        "physio_covariance_auc": physio["physio_covariance_auc"],
        "physio_ready_for_training": physio["physio_ready_for_training"],
        "covariance_detector_is_construct_specific": physio["covariance_detector_is_construct_specific"],
        "protocol_leakage_detected": bool(protocol.get("protocol_leakage_detected", False)) or bool(layer3.get("protocol_leakage_detected", False)) or bool(physio.get("protocol_leakage_detected", False)),
        "decision": decision,
        "next_action": next_action(decision, physio),
    }
    (outputs / "compact_result_v5.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    write_report(outputs / "SAS_CERT_CBRAMOD_V5_LOCKED_CONFIRMATORY_REPORT.md", compact, layer3, physio, v4, raw, ls, artifact_raw)
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def find_pair(rows: Sequence[Dict[str, object]], group: str, baseline: str) -> Dict[str, object]:
    for row in rows:
        if row["group"] == group and row["baseline_group"] == baseline:
            return row
    raise KeyError(f"Missing pair {group} vs {baseline}")


def decide(raw: Dict[str, object], ls: Dict[str, object]) -> str:
    raw_go = raw["final_group_decision"] == "GO_RAW_BASELINE"
    ls_go = ls["final_group_decision"] == "GO_LS_BASELINE"
    if raw_go and ls_go:
        return "GO_LOCKED_METHOD_STABLE"
    if raw_go and not ls_go:
        return "LABEL_SMOOTHING_CONFOUNDED"
    if raw["passed_macro_f1"] and not raw["passed_acc"]:
        return "PARTIAL_GO_F1_STABLE"
    if not raw["passed_ece"] or not ls["passed_ece"]:
        return "CALIBRATION_REGRESSION"
    return "EFFECT_NOT_STABLE"


def next_action(decision: str, physio: Dict[str, object]) -> str:
    if decision == "GO_LOCKED_METHOD_STABLE":
        action = "enter BCI IV-2b small extension or longer epoch shadow"
    elif decision == "LABEL_SMOOTHING_CONFOUNDED":
        action = "keep NaiveAug_LS010 as all future main baseline"
    elif decision == "PARTIAL_GO_F1_STABLE":
        action = "emphasize Macro-F1 and reliable adaptation as primary paper metric"
    elif decision == "CALIBRATION_REGRESSION":
        action = "return to v3 and inspect whether seed expansion changed calibration"
    else:
        action = "do not expand; return to support split stability analysis"
    if physio.get("covariance_detector_is_construct_specific"):
        action += "; keep physio_covariance out of training weights"
    return action


def write_report(path: Path, compact: Dict[str, object], layer3: Dict[str, object], physio: Dict[str, object], v4: Dict[str, object], raw: Dict[str, object], ls: Dict[str, object], artifact_raw: Dict[str, object]) -> None:
    lines = [
        "# SAS-Cert-CBraMod V5 Locked Confirmatory Report",
        "",
        "## Summary",
        "",
        f"- Status: `{compact['status']}`",
        f"- Main method: `{compact['main_method']}`",
        f"- Decision: `{compact['decision']}`",
        f"- Protocol leakage detected: `{compact['protocol_leakage_detected']}`",
        "",
        "## Why V4 Was GO",
        "",
        f"- V4 Acc/Macro-F1 deltas were `{v4['delta_acc']:+.4f}` / `{v4['delta_macro_f1']:+.4f}`.",
        f"- V4 ECE/NLL/Brier deltas were `{v4['delta_ece']:+.4f}` / `{v4['delta_nll']:+.4f}` / `{v4['delta_brier']:+.4f}`.",
        f"- V4 subject win rates were `{v4['subject_win_rate_acc']:.4f}` / `{v4['subject_win_rate_macro_f1']:.4f}`.",
        "",
        "## Locked Method Vs Baselines",
        "",
        "| Baseline | Acc | Macro-F1 | ECE | NLL | Brier | Subject Win Acc/F1 | Seed Win Acc/F1 | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        pair_line("NaiveAug_raw", raw),
        pair_line("NaiveAug_LS010", ls),
        "",
        "## Label Smoothing Confound Check",
        "",
        f"- Raw baseline decision: `{raw['final_group_decision']}`",
        f"- LS baseline decision: `{ls['final_group_decision']}`",
        "- If the LS baseline fails while raw passes, the report must treat the result as label-smoothing-confounded.",
        "",
        "## Artifact Diagnostic Branch",
        "",
        f"- ArtifactReject_raw_diagnostic vs NaiveAug_raw Acc/Macro-F1: `{artifact_raw['delta_acc']:+.4f}` / `{artifact_raw['delta_macro_f1']:+.4f}`.",
        f"- ECE/NLL/Brier: `{artifact_raw['delta_ece']:+.4f}` / `{artifact_raw['delta_nll']:+.4f}` / `{artifact_raw['delta_brier']:+.4f}`.",
        f"- Decision: `{artifact_raw['final_group_decision']}`.",
        "",
        "## Physio Covariance Audit",
        "",
        f"- physio_covariance_auc: `{compact['physio_covariance_auc']:.4f}`",
        f"- covariance_detector_is_construct_specific: `{compact['covariance_detector_is_construct_specific']}`",
        f"- physio_ready_for_training: `{compact['physio_ready_for_training']}`",
        f"- conclusion: `{physio['conclusion']}`",
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
    path.write_text("\n".join(lines) + "\n")


def pair_line(name: str, row: Dict[str, object]) -> str:
    return (
        f"| {name} | {row['delta_acc']:+.4f} | {row['delta_macro_f1']:+.4f} | {row['delta_ece']:+.4f} | "
        f"{row['delta_nll']:+.4f} | {row['delta_brier']:+.4f} | "
        f"{row['subject_win_rate_acc']:.4f}/{row['subject_win_rate_macro_f1']:.4f} | "
        f"{row['seed_win_rate_acc']:.4f}/{row['seed_win_rate_macro_f1']:.4f} | {row['final_group_decision']} |"
    )


if __name__ == "__main__":
    main()
