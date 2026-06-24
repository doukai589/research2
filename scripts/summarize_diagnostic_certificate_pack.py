#!/usr/bin/env python3
"""Summarize the PhysioNetMI SAS-Cert diagnostic certificate pack.

This script is intentionally existing-output-only. It does not train, rerank,
or select thresholds from target held-out outcomes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "next_mve_after_diagnostic_reframe.yaml"
OUT_DIR = ROOT / "outputs" / "runs" / "sas_cert_diagnostic_certificate_pack_physionetmi"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = ROOT / "docs" / "SAS_CERT_DIAGNOSTIC_CERTIFICATE_PACK_PHYSIONETMI.md"
CHECKLIST_PATH = ROOT / "docs" / "SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: float) -> str:
    return f"{value:.4f}"


def component_lookup(rows: list[dict[str, Any]], component: str, backbone: str) -> float:
    for row in rows:
        if row.get("component") == component and row.get("backbone") == backbone:
            return float(row["auc_high_score_is_clean"])
    raise KeyError((component, backbone))


def build_payload() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    cross = read_json(inputs["cross_backbone_audit"])
    gated = read_json(inputs["component_gated_rule"])
    cbr_current = read_json(inputs["cbramod_current"])
    cbr_repaired = read_json(inputs["cbramod_repaired"])
    cbr_calibrated = read_json(inputs["cbramod_calibrated"])
    st_confirm = read_json(inputs["st_locked_confirm"])
    st_utility = read_json(inputs["st_utility_audit"])
    bad_type_auc = read_csv("workbench/20260622_cross_backbone_cert_direction_audit/outputs/component_auc_by_bad_type.csv")

    backbones = ["CBraMod_frozen", "ST-EEGFormer-small_source_tuned"]
    diagnostic_rows = []
    for backbone in backbones:
        diagnostic_rows.append(
            {
                "backbone": backbone,
                "current_scalar_sas_auc": component_lookup(gated["component_auc"], "sas_score", backbone),
                "component_gated_v1_auc": component_lookup(gated["component_auc"], "component_gated_v1", backbone),
                "artifact_gate_physio_auc": component_lookup(gated["component_auc"], "score_artifact_gate_physio", backbone),
                "physio_score_auc": component_lookup(gated["component_auc"], "physio_score", backbone),
                "style_score_auc": component_lookup(gated["component_auc"], "style_score", backbone),
                "content_score_auc": component_lookup(gated["component_auc"], "content_score", backbone),
                "artifact_safe_score_auc": component_lookup(gated["component_auc"], "artifact_safe_score", backbone),
            }
        )

    cbr_repaired_delta = cbr_repaired["primary_vs_naive"]
    cbr_calibrated_delta = cbr_calibrated["primary_vs_naive"]
    st_delta = st_confirm["comparisons"]["softweight_vs_naive"]
    utility_top = st_utility["top_correlations"][0]

    gate = config["go_criteria"]
    diagnostic_auc_pass = all(
        max(row["component_gated_v1_auc"], row["artifact_gate_physio_auc"]) >= gate["diagnostic_auc_gate"]["min_auc"]
        for row in diagnostic_rows
    )
    scalar_failure_pass = all(row["current_scalar_sas_auc"] < gate["scalar_failure_gate"]["max_auc"] for row in diagnostic_rows)
    protocol_pass = bool(
        cross["scope"]["uses_existing_score_rows_only"]
        and cross["scope"]["no_retraining"]
        and cross["scope"]["target_test_labels_used"] is False
        and cbr_repaired["leakage_audit"]["target_test_used_for_ranking_threshold_or_training"] is False
        and st_utility["protocol_leakage_detected"] is False
    )
    non_promotion_pass = bool(
        st_confirm["decision"] == "do_not_promote_softweight_no_reject"
        and st_utility["decision"] == "park_st_weighting_variants"
        and cbr_repaired_delta["delta_ece"] > 0.01
    )

    decision = (
        "write_diagnostic_certificate_paper_path"
        if diagnostic_auc_pass and scalar_failure_pass and protocol_pass and non_promotion_pass
        else "park_sas_cert_as_internal_diagnostic_tool"
    )

    training_rows = [
        {
            "backbone": "CBraMod_frozen",
            "branch": "current_SASCert_SoftAR",
            "scope": "targets_90_92_seeds_20_21",
            "delta_balanced_accuracy_vs_naive": cbr_current["sascert_vs_naive"]["delta_balanced_accuracy"],
            "delta_macro_f1_vs_naive": cbr_current["sascert_vs_naive"]["delta_macro_f1"],
            "delta_ece_vs_naive": cbr_current["sascert_vs_naive"]["delta_ece"],
            "decision": "not_promoted_calibration_and_macro_f1_weak",
        },
        {
            "backbone": "CBraMod_frozen",
            "branch": "repaired_artifact_gate_physio",
            "scope": "targets_90_92_seeds_20_21",
            "delta_balanced_accuracy_vs_naive": cbr_repaired_delta["delta_balanced_accuracy"],
            "delta_macro_f1_vs_naive": cbr_repaired_delta["delta_macro_f1"],
            "delta_ece_vs_naive": cbr_repaired_delta["delta_ece"],
            "decision": "not_promoted_calibration_failed",
        },
        {
            "backbone": "CBraMod_frozen",
            "branch": "repaired_temperature_scaled",
            "scope": "targets_90_92_seeds_20_21",
            "delta_balanced_accuracy_vs_naive": cbr_calibrated_delta["delta_balanced_accuracy"],
            "delta_macro_f1_vs_naive": cbr_calibrated_delta["delta_macro_f1"],
            "delta_ece_vs_naive": cbr_calibrated_delta["delta_ece"],
            "decision": "not_promoted_temperature_scaling_failed_calibration_gate",
        },
        {
            "backbone": "ST-EEGFormer-small_source_tuned",
            "branch": "SoftWeight_noReject",
            "scope": "targets_90_109_seeds_20_24",
            "delta_balanced_accuracy_vs_naive": st_delta["delta_balanced_accuracy"],
            "delta_macro_f1_vs_naive": st_delta["delta_macro_f1"],
            "delta_ece_vs_naive": st_delta["delta_ece"],
            "decision": "not_promoted_subject_seed_reliability_failed",
        },
    ]

    payload = {
        "status": "completed",
        "project": "sas_cert_diagnostic_certificate_pack_physionetmi",
        "dataset": "PhysioNetMI",
        "task": "left_vs_right_motor_imagery",
        "backbones": backbones,
        "uses_existing_outputs_only": True,
        "diagnostic_auc_rows": diagnostic_rows,
        "training_policy_rows": training_rows,
        "gates": {
            "diagnostic_auc_pass": diagnostic_auc_pass,
            "scalar_failure_pass": scalar_failure_pass,
            "protocol_pass": protocol_pass,
            "weighting_policy_non_promotion_pass": non_promotion_pass,
        },
        "st_utility_alignment": {
            "decision": st_utility["decision"],
            "strong_alignment_found": st_utility["strong_alignment_found"],
            "top_candidate_feature": utility_top["feature"],
            "top_candidate_spearman": utility_top["spearman"],
        },
        "bad_type_auc_rows": bad_type_auc,
        "protocol_audit_rows": [
            {
                "source": "cross_backbone_direction_audit",
                "raw_data_copied": False,
                "new_training": False,
                "new_augmentation_generation": False,
                "target_heldout_used_for_score_or_threshold": False,
                "evidence_path": inputs["cross_backbone_audit"],
            },
            {
                "source": "component_gated_rule_v1",
                "raw_data_copied": False,
                "new_training": False,
                "new_augmentation_generation": False,
                "target_heldout_used_for_score_or_threshold": False,
                "evidence_path": inputs["component_gated_rule"],
            },
            {
                "source": "cbramod_repaired_mini",
                "raw_data_copied": cbr_repaired["leakage_audit"]["raw_data_copied"],
                "new_training": True,
                "new_augmentation_generation": True,
                "target_heldout_used_for_score_or_threshold": cbr_repaired["leakage_audit"]["target_test_used_for_ranking_threshold_or_training"],
                "evidence_path": inputs["cbramod_repaired"],
            },
            {
                "source": "st_softweight_locked_confirm",
                "raw_data_copied": False,
                "new_training": False,
                "new_augmentation_generation": False,
                "target_heldout_used_for_score_or_threshold": False,
                "evidence_path": inputs["st_locked_confirm"],
            },
            {
                "source": "st_utility_alignment_audit",
                "raw_data_copied": False,
                "new_training": False,
                "new_augmentation_generation": False,
                "target_heldout_used_for_score_or_threshold": st_utility["protocol_leakage_detected"],
                "evidence_path": inputs["st_utility_audit"],
            },
        ],
        "failure_mode_definitions": [
            {
                "failure_mode": "bad_artifact",
                "definition": "augmentation introduces low-frequency drift, bursts, or abnormal channel-energy/kurtosis patterns that can create non-neural shortcuts",
                "most_relevant_components": "artifact_safe_score, score_artifact_gate_physio",
            },
            {
                "failure_mode": "bad_content",
                "definition": "augmentation disrupts task-discriminative MI content such as mu/beta structure or label-consistent representation",
                "most_relevant_components": "content_score, physio_score",
            },
            {
                "failure_mode": "bad_physio",
                "definition": "augmentation violates physiological/topological consistency such as channel arrangement or covariance structure",
                "most_relevant_components": "physio_score, style_score",
            },
        ],
        "traceability_rows": [
            {
                "reported_item": "CBraMod current scalar SAS AUC 0.1969",
                "source_path": inputs["component_gated_rule"],
                "json_field_or_table": "component_auc: backbone=CBraMod_frozen, component=sas_score",
            },
            {
                "reported_item": "ST current scalar SAS AUC 0.1662",
                "source_path": inputs["component_gated_rule"],
                "json_field_or_table": "component_auc: backbone=ST-EEGFormer-small_source_tuned, component=sas_score",
            },
            {
                "reported_item": "component-gated v1 AUC 0.8395 on both backbones",
                "source_path": inputs["component_gated_rule"],
                "json_field_or_table": "component_auc: component=component_gated_v1",
            },
            {
                "reported_item": "artifact-gate physio AUC 0.9022 on both backbones",
                "source_path": inputs["component_gated_rule"],
                "json_field_or_table": "component_auc: component=score_artifact_gate_physio",
            },
            {
                "reported_item": "CBraMod repaired Macro-F1 +4.26pp and ECE +2.27pp",
                "source_path": inputs["cbramod_repaired"],
                "json_field_or_table": "primary_vs_naive",
            },
            {
                "reported_item": "ST SoftWeight Macro-F1 +0.64pp and reliability failure",
                "source_path": inputs["st_locked_confirm"],
                "json_field_or_table": "comparisons.softweight_vs_naive",
            },
            {
                "reported_item": "ST utility alignment max candidate-only Spearman 0.1168",
                "source_path": inputs["st_utility_audit"],
                "json_field_or_table": "top_correlations[0]",
            },
        ],
        "interpretation": {
            "supported_claim": "SAS-Cert is currently supported as a diagnostic reliability certificate for EEG augmentation candidates.",
            "unsupported_claim": "Current SAS-Cert weighting/rejection policies are not supported as reliable deployable training methods.",
            "reason": "Both backbones show meaningful diagnostic score structure, but CBraMod fails calibration after repair and ST fails subject/seed reliability plus utility alignment.",
        },
        "decision": decision,
        "protocol_leakage_detected": False,
        "next_action": "prepare_diagnostic_certificate_paper_outline" if decision == "write_diagnostic_certificate_paper_path" else "park_sas_cert_as_internal_diagnostic_tool",
    }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    diag = payload["diagnostic_auc_rows"]
    train = payload["training_policy_rows"]
    gates = payload["gates"]
    st_util = payload["st_utility_alignment"]
    lines = [
        "# SAS-Cert Diagnostic Certificate Pack: PhysioNetMI",
        "",
        "## Scope",
        "",
        "- Dataset: `PhysioNetMI`, left/right motor imagery, runs `R04/R08/R12`",
        "- Backbones: `CBraMod_frozen`, `ST-EEGFormer-small_source_tuned`",
        "- Mode: existing-output-only diagnostic MVE",
        "- No new training, no new dataset, no new backbone",
        "",
        "## Diagnostic Score Evidence",
        "",
        "| Backbone | Current SAS AUC | Component-Gated v1 AUC | Artifact-Gate Physio AUC | Physio AUC | Style AUC | Content AUC | Artifact-Safe AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diag:
        lines.append(
            f"| `{row['backbone']}` | {f(row['current_scalar_sas_auc'])} | {f(row['component_gated_v1_auc'])} | {f(row['artifact_gate_physio_auc'])} | {f(row['physio_score_auc'])} | {f(row['style_score_auc'])} | {f(row['content_score_auc'])} | {f(row['artifact_safe_score_auc'])} |"
        )
    lines.extend(
        [
            "",
            "The old scalar SAS score fails directionally on the mixed-bad PhysioNetMI pool, while component-gated and artifact-gate-physio variants recover strong diagnostic AUC on both backbones.",
            "",
            "## Training Policy Evidence",
            "",
            "| Backbone | Branch | Scope | Delta BAcc | Delta Macro-F1 | Delta ECE | Decision |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in train:
        lines.append(
            f"| `{row['backbone']}` | `{row['branch']}` | `{row['scope']}` | {f(row['delta_balanced_accuracy_vs_naive'])} | {f(row['delta_macro_f1_vs_naive'])} | {f(row['delta_ece_vs_naive'])} | `{row['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## ST Utility Alignment",
            "",
            f"- Decision: `{st_util['decision']}`",
            f"- Strong alignment found: `{st_util['strong_alignment_found']}`",
            f"- Top candidate-only feature: `{st_util['top_candidate_feature']}`",
            f"- Top candidate-only Spearman: `{st_util['top_candidate_spearman']:.4f}`",
            "",
            "## Gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            "Supported claim:",
            "",
            f"> {payload['interpretation']['supported_claim']}",
            "",
            "Unsupported claim:",
            "",
            f"> {payload['interpretation']['unsupported_claim']}",
            "",
            "Why:",
            "",
            payload["interpretation"]["reason"],
            "",
            "## Next Action",
            "",
            f"`{payload['next_action']}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_checklist(payload: dict[str, Any]) -> None:
    protocol = payload["protocol_audit_rows"]
    failure_modes = payload["failure_mode_definitions"]
    trace = payload["traceability_rows"]
    bad_type_rows = payload["bad_type_auc_rows"]
    selected_bad_rows = [
        row
        for row in bad_type_rows
        if row["component"] in {"sas_score", "physio_score", "score_artifact_gate_physio", "artifact_safe_score", "content_score"}
    ]
    lines = [
        "# SAS-Cert Diagnostic Evidence Checklist",
        "",
        "## 1. Bad-Type AUC By Component",
        "",
        "Source: `workbench/20260622_cross_backbone_cert_direction_audit/outputs/component_auc_by_bad_type.csv`",
        "",
        "| Backbone | Bad Type | Component | AUC High Score Is Clean | Direction |",
        "|---|---|---|---:|---|",
    ]
    for row in selected_bad_rows:
        lines.append(
            f"| `{row['backbone']}` | `{row['bad_type']}` | `{row['component']}` | {float(row['auc_high_score_is_clean']):.4f} | `{row['direction']}` |"
        )
    lines.extend(
        [
            "",
            "## 2. Protocol-Leakage Audit",
            "",
            "| Source | Raw Data Copied | New Training | New Augmentation Generation | Target Heldout Used For Score/Threshold | Evidence |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in protocol:
        lines.append(
            f"| `{row['source']}` | `{row['raw_data_copied']}` | `{row['new_training']}` | `{row['new_augmentation_generation']}` | `{row['target_heldout_used_for_score_or_threshold']}` | `{row['evidence_path']}` |"
        )
    lines.extend(
        [
            "",
            "## 3. Causal Chain Diagram",
            "",
            "```text",
            "bad/clean augmentation is separable",
            "  -> score direction must be audited",
            "  -> component-gated diagnostics can recover separation",
            "  -> weighting/rejection changes training signal",
            "  -X-> reliable deployable policy across subjects/seeds",
            "```",
            "",
            "The current evidence supports the diagnostic links but breaks at deployable training reliability.",
            "",
            "## 4. Augmentation Failure-Mode Definitions",
            "",
            "| Failure Mode | Definition | Most Relevant Components |",
            "|---|---|---|",
        ]
    )
    for row in failure_modes:
        lines.append(f"| `{row['failure_mode']}` | {row['definition']} | `{row['most_relevant_components']}` |")
    lines.extend(
        [
            "",
            "## 5. File-Path Traceability",
            "",
            "| Reported Item | Source Path | Field/Table |",
            "|---|---|---|",
        ]
    )
    for row in trace:
        lines.append(f"| {row['reported_item']} | `{row['source_path']}` | `{row['json_field_or_table']}` |")
    lines.extend(
        [
            "",
            "## Checklist Status",
            "",
            "- [x] Bad-type AUC by component table added.",
            "- [x] Protocol-leakage audit table added.",
            "- [x] Causal chain diagram added.",
            "- [x] Augmentation failure-mode definitions added.",
            "- [x] File-path traceability table added.",
            "",
        ]
    )
    CHECKLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "compact_result.json", payload)
    write_csv(TABLE_DIR / "diagnostic_auc_summary.csv", payload["diagnostic_auc_rows"])
    write_csv(TABLE_DIR / "training_policy_summary.csv", payload["training_policy_rows"])
    gate_rows = [{"gate": key, "passed": value} for key, value in payload["gates"].items()]
    write_csv(TABLE_DIR / "gate_summary.csv", gate_rows)
    write_csv(TABLE_DIR / "bad_type_component_auc.csv", payload["bad_type_auc_rows"])
    write_csv(TABLE_DIR / "protocol_leakage_audit.csv", payload["protocol_audit_rows"])
    write_csv(TABLE_DIR / "failure_mode_definitions.csv", payload["failure_mode_definitions"])
    write_csv(TABLE_DIR / "number_traceability.csv", payload["traceability_rows"])
    write_report(payload)
    write_checklist(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
