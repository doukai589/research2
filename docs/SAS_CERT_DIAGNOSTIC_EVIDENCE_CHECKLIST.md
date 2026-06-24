# SAS-Cert Diagnostic Evidence Checklist

## 1. Bad-Type AUC By Component

Source: `workbench/20260622_cross_backbone_cert_direction_audit/outputs/component_auc_by_bad_type.csv`

| Backbone | Bad Type | Component | AUC High Score Is Clean | Direction |
|---|---|---|---:|---|
| `CBraMod_frozen` | `bad_artifact` | `content_score` | 0.0054 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_artifact` | `physio_score` | 0.6531 | `clean_high` |
| `CBraMod_frozen` | `bad_artifact` | `artifact_safe_score` | 1.0000 | `clean_high` |
| `CBraMod_frozen` | `bad_artifact` | `sas_score` | 0.5085 | `weak_or_mixed` |
| `CBraMod_frozen` | `bad_artifact` | `score_artifact_gate_physio` | 0.8264 | `clean_high` |
| `CBraMod_frozen` | `bad_content` | `content_score` | 0.9025 | `clean_high` |
| `CBraMod_frozen` | `bad_content` | `physio_score` | 0.9407 | `clean_high` |
| `CBraMod_frozen` | `bad_content` | `artifact_safe_score` | 0.0000 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_content` | `sas_score` | 0.0819 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_content` | `score_artifact_gate_physio` | 0.9407 | `clean_high` |
| `CBraMod_frozen` | `bad_physio` | `content_score` | 0.0054 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_physio` | `physio_score` | 0.9396 | `clean_high` |
| `CBraMod_frozen` | `bad_physio` | `artifact_safe_score` | 0.0000 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_physio` | `sas_score` | 0.0003 | `bad_high_or_inverted` |
| `CBraMod_frozen` | `bad_physio` | `score_artifact_gate_physio` | 0.9396 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `content_score` | 0.0313 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `physio_score` | 0.6531 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `artifact_safe_score` | 1.0000 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `sas_score` | 0.3058 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `score_artifact_gate_physio` | 0.8264 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `content_score` | 0.2630 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `physio_score` | 0.9407 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `artifact_safe_score` | 0.0000 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `sas_score` | 0.0183 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `score_artifact_gate_physio` | 0.9407 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `content_score` | 0.5224 | `weak_or_mixed` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `physio_score` | 0.9396 | `clean_high` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `artifact_safe_score` | 0.0000 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `sas_score` | 0.1746 | `bad_high_or_inverted` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `score_artifact_gate_physio` | 0.9396 | `clean_high` |

## 2. Protocol-Leakage Audit

| Source | Raw Data Copied | New Training | New Augmentation Generation | Target Heldout Used For Score/Threshold | Evidence |
|---|---:|---:|---:|---:|---|
| `cross_backbone_direction_audit` | `False` | `False` | `False` | `False` | `workbench/20260622_cross_backbone_cert_direction_audit/outputs/compact_cross_backbone_cert_direction_audit.json` |
| `component_gated_rule_v1` | `False` | `False` | `False` | `False` | `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json` |
| `cbramod_repaired_mini` | `False` | `True` | `True` | `False` | `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_repaired_mini.json` |
| `st_softweight_locked_confirm` | `False` | `False` | `False` | `False` | `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json` |
| `st_utility_alignment_audit` | `False` | `False` | `False` | `False` | `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/compact_utility_alignment_audit.json` |

## 3. Causal Chain Diagram

```text
bad/clean augmentation is separable
  -> score direction must be audited
  -> component-gated diagnostics can recover separation
  -> weighting/rejection changes training signal
  -X-> reliable deployable policy across subjects/seeds
```

The current evidence supports the diagnostic links but breaks at deployable training reliability.

## 4. Augmentation Failure-Mode Definitions

| Failure Mode | Definition | Most Relevant Components |
|---|---|---|
| `bad_artifact` | augmentation introduces low-frequency drift, bursts, or abnormal channel-energy/kurtosis patterns that can create non-neural shortcuts | `artifact_safe_score, score_artifact_gate_physio` |
| `bad_content` | augmentation disrupts task-discriminative MI content such as mu/beta structure or label-consistent representation | `content_score, physio_score` |
| `bad_physio` | augmentation violates physiological/topological consistency such as channel arrangement or covariance structure | `physio_score, style_score` |

## 5. File-Path Traceability

| Reported Item | Source Path | Field/Table |
|---|---|---|
| CBraMod current scalar SAS AUC 0.1969 | `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json` | `component_auc: backbone=CBraMod_frozen, component=sas_score` |
| ST current scalar SAS AUC 0.1662 | `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json` | `component_auc: backbone=ST-EEGFormer-small_source_tuned, component=sas_score` |
| component-gated v1 AUC 0.8395 on both backbones | `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json` | `component_auc: component=component_gated_v1` |
| artifact-gate physio AUC 0.9022 on both backbones | `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json` | `component_auc: component=score_artifact_gate_physio` |
| CBraMod repaired Macro-F1 +4.26pp and ECE +2.27pp | `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_repaired_mini.json` | `primary_vs_naive` |
| ST SoftWeight Macro-F1 +0.64pp and reliability failure | `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json` | `comparisons.softweight_vs_naive` |
| ST utility alignment max candidate-only Spearman 0.1168 | `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/compact_utility_alignment_audit.json` | `top_correlations[0]` |

## Checklist Status

- [x] Bad-type AUC by component table added.
- [x] Protocol-leakage audit table added.
- [x] Causal chain diagram added.
- [x] Augmentation failure-mode definitions added.
- [x] File-path traceability table added.
