# Appendix. Protocol Leakage Audit

| source | raw_data_copied | new_training | new_augmentation_generation | target_heldout_used_for_score_or_threshold | evidence_path |
| --- | --- | --- | --- | --- | --- |
| cross_backbone_direction_audit | False | False | False | False | workbench/20260622_cross_backbone_cert_direction_audit/outputs/compact_cross_backbone_cert_direction_audit.json |
| component_gated_rule_v1 | False | False | False | False | workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json |
| cbramod_repaired_mini | False | True | True | False | workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_repaired_mini.json |
| st_softweight_locked_confirm | False | False | False | False | workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json |
| st_utility_alignment_audit | False | False | False | False | workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/compact_utility_alignment_audit.json |
