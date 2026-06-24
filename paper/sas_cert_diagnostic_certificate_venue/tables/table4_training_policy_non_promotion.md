# Table 4. Training Policy Non-Promotion

| backbone | branch | scope | delta_balanced_accuracy_vs_naive | delta_macro_f1_vs_naive | delta_ece_vs_naive | decision |
| --- | --- | --- | --- | --- | --- | --- |
| CBraMod_frozen | current_SASCert_SoftAR | targets_90_92_seeds_20_21 | 0.0124 | 0.0005 | 0.0277 | not_promoted_calibration_and_macro_f1_weak |
| CBraMod_frozen | repaired_artifact_gate_physio | targets_90_92_seeds_20_21 | 0.0111 | 0.0426 | 0.0227 | not_promoted_calibration_failed |
| CBraMod_frozen | repaired_temperature_scaled | targets_90_92_seeds_20_21 | 0.0111 | 0.0426 | 0.0221 | not_promoted_temperature_scaling_failed_calibration_gate |
| ST-EEGFormer-small_source_tuned | SoftWeight_noReject | targets_90_109_seeds_20_24 | 0.0065 | 0.0064 | 0.0003 | not_promoted_subject_seed_reliability_failed |
