from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"


def main() -> None:
    layer2_v2 = read_json(OUTPUTS / "layer2_v2" / "layer2_v2_summary.json")
    layer3_v2 = read_json(OUTPUTS / "layer3_v2" / "layer3_v2_summary.json")
    layer2 = read_json(OUTPUTS / "layer2" / "layer2_summary.json")
    layer3 = read_json(OUTPUTS / "layer3" / "layer3_summary.json")
    protocol = read_json(OUTPUTS / "audit" / "protocol_audit.json")

    swar_acc = paired_delta(layer3_v2["paired_comparison"], "SoftWeightArtifactReject_minus_NaiveAug", "acc")
    swar_f1 = paired_delta(layer3_v2["paired_comparison"], "SoftWeightArtifactReject_minus_NaiveAug", "macro_f1")
    swar_ece = paired_delta(layer3_v2["paired_comparison"], "SoftWeightArtifactReject_minus_NaiveAug", "ece")
    swar_vs_random = paired_delta(layer3_v2["paired_comparison"], "SoftWeightArtifactReject_minus_Random50", "acc")
    swar_vs_top50 = paired_delta(layer3_v2["paired_comparison"], "SoftWeightArtifactReject_minus_SASCertTop50", "acc")
    artifact_acc = paired_delta(layer3_v2["paired_comparison"], "ArtifactReject_minus_NaiveAug", "acc")
    artifact_f1 = paired_delta(layer3_v2["paired_comparison"], "ArtifactReject_minus_NaiveAug", "macro_f1")
    artifact_ece = paired_delta(layer3_v2["paired_comparison"], "ArtifactReject_minus_NaiveAug", "ece")

    decisions = []
    if (swar_acc >= 0.01 or swar_f1 >= 0.01) and swar_ece <= 0.01:
        decisions.append("GO_SOFTWEIGHT")
    if layer2_v2["physio_original_auc"] < 0.5 or layer2_v2["physio_fixed_badphysio_auc"] < 0.6:
        decisions.append("REFINE_PHYSIO")
    if layer3["sas_top50_minus_naive_acc"] <= 0:
        decisions.append("STOP_HARD_TOP50")
    artifact_calibration_risk = artifact_ece > 0.01
    if layer2["bad_artifact_auc"] >= 0.90 and artifact_acc > 0 and artifact_calibration_risk:
        decisions.append("ARTIFACT_ONLY_BRANCH")
    if not decisions:
        decisions.append("NO_GO")

    compact = {
        "project": "sas_cert_cbramod_mve",
        "version": "v2_softweight_artifact_reject",
        "status": "completed",
        "best_score_variant": layer3_v2["selected_score_variant"],
        "physio_original_auc": layer2_v2["physio_original_auc"],
        "physio_fixed_used": layer2_v2["physio_fixed_used"],
        "physio_fixed_badphysio_auc": layer2_v2["physio_fixed_badphysio_auc"],
        "softweight_artifactreject_minus_naive": {
            "acc": swar_acc,
            "macro_f1": swar_f1,
            "ece": swar_ece,
        },
        "softweight_artifactreject_minus_random50_acc": swar_vs_random,
        "softweight_artifactreject_minus_sascerttop50_acc": swar_vs_top50,
        "artifactreject_minus_naive": {
            "acc": artifact_acc,
            "macro_f1": artifact_f1,
            "ece": artifact_ece,
            "calibration_risk": artifact_calibration_risk,
        },
        "hard_top50_minus_naive_acc": layer3["sas_top50_minus_naive_acc"],
        "layer2_v2_overall_auc_by_variant": layer2_v2["variant_overall_auc"],
        "final_decision": decisions,
        "protocol_leakage_detected": bool(protocol.get("protocol_leakage_detected", False)),
        "next_action": next_action(decisions),
    }
    write_json(OUTPUTS / "compact_result_v2.json", compact)
    write_report(OUTPUTS / "SAS_CERT_CBRAMOD_V2_REPORT.md", compact, layer2, layer3, layer2_v2, layer3_v2)
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def paired_delta(paired: Sequence[Dict[str, object]], comparison: str, metric: str) -> float:
    for row in paired:
        if row["comparison"] == comparison and row["metric"] == metric:
            return float(row["mean_delta"])
    return float("nan")


def next_action(decisions: Sequence[str]) -> str:
    actions = []
    if "GO_SOFTWEIGHT" in decisions:
        actions.append("run longer epochs or more seeds with SoftWeightArtifactReject as preregistered main group")
    if "REFINE_PHYSIO" in decisions:
        actions.append("split BadPhysio into bandpower/covariance/topology/channel_order diagnostics")
    if "ARTIFACT_ONLY_BRANCH" in decisions:
        actions.append("test calibration-aware loss or temperature scaling for artifact-focused branch")
    if "STOP_HARD_TOP50" in decisions:
        actions.append("stop treating hard SASCertTop50 as main method")
    return "; ".join(actions) if actions else "inspect v2 diagnostics before expansion"


def write_report(path: Path, compact: Dict[str, object], layer2: Dict[str, object], layer3: Dict[str, object], layer2_v2: Dict[str, object], layer3_v2: Dict[str, object]) -> None:
    artifact = compact["artifactreject_minus_naive"]
    swar = compact["softweight_artifactreject_minus_naive"]
    lines = [
        "# SAS-Cert-CBraMod V2 SoftWeight ArtifactReject Report",
        "",
        "## Summary",
        "",
        f"- Best score variant: `{compact['best_score_variant']}`",
        f"- Physio original AUC: `{compact['physio_original_auc']:.4f}`",
        f"- Physio fixed used: `{compact['physio_fixed_used']}`",
        f"- Final decision: `{', '.join(compact['final_decision'])}`",
        f"- Protocol leakage detected: `{compact['protocol_leakage_detected']}`",
        "",
        "## Why Hard Top50 Failed",
        "",
        f"- Layer 2 original overall AUC was `{layer2['overall_auc']:.4f}`, below the 0.70 gate.",
        f"- SASCertTop50 vs NaiveAug Acc was `{layer3['sas_top50_minus_naive_acc']:+.4f}`.",
        f"- SASCertTop50 vs Random50 Acc was `{layer3['sas_top50_minus_random50_acc']:+.4f}`, so hard Top50 only beat random selection weakly and did not beat using all augmentations.",
        "",
        "## Layer 2-v2 Diagnosis",
        "",
        "| Variant | Overall clean-vs-bad AUC |",
        "| --- | ---: |",
    ]
    for name, auc in sorted(layer2_v2["variant_overall_auc"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {name} | {auc:.4f} |")
    lines.extend(
        [
            "",
            f"- Physio direction maybe wrong: `{layer2_v2['physio_direction_maybe_wrong']}`",
            f"- Physio fixed adopted: `{layer2_v2['physio_fixed_used']}`",
            f"- Physio fixed BadPhysio AUC: `{layer2_v2['physio_fixed_badphysio_auc']:.4f}`",
            "",
            "## Layer 3-v2 Training",
            "",
            "| Comparison | Acc delta | Macro-F1 delta | ECE delta |",
            "| --- | ---: | ---: | ---: |",
            f"| SoftWeightArtifactReject - NaiveAug | {swar['acc']:+.4f} | {swar['macro_f1']:+.4f} | {swar['ece']:+.4f} |",
            f"| ArtifactReject - NaiveAug | {artifact['acc']:+.4f} | {artifact['macro_f1']:+.4f} | {artifact['ece']:+.4f} |",
            "",
            "## ArtifactReject Calibration Risk",
            "",
            f"- ArtifactReject calibration risk: `{artifact['calibration_risk']}`",
            f"- It is not the main method when ECE delta exceeds `+0.01`; observed ECE delta is `{artifact['ece']:+.4f}`.",
            "",
            "## Decision Rules",
            "",
            f"- GO_SOFTWEIGHT: `{ 'GO_SOFTWEIGHT' in compact['final_decision'] }`",
            f"- REFINE_PHYSIO: `{ 'REFINE_PHYSIO' in compact['final_decision'] }`",
            f"- STOP_HARD_TOP50: `{ 'STOP_HARD_TOP50' in compact['final_decision'] }`",
            f"- ARTIFACT_ONLY_BRANCH: `{ 'ARTIFACT_ONLY_BRANCH' in compact['final_decision'] }`",
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
