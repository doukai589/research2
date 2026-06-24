# SAS-Cert-SoftAR-LS v1.1 ST-EEGFormer PhysioNetMI Report

## Scope

- Targets: `[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]`
- Seeds: `[20, 21, 22, 23, 24]`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `v1_1_source_tuned_full`
- Source-tuned checkpoint: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Candidate augmentations per support trial: `6`
- Artifact reject percentile: `90.0`

## Main Result

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7156 | 0.7098 | 0.7825 | 0.2093 | 0.7208 | 0.4225 | 0.0000 | 1.0000 |
| ArtifactReject_LS010 | 0.7074 | 0.7013 | 0.7823 | 0.2150 | 0.7304 | 0.4267 | 0.1000 | 1.0000 |
| SoftWeight_noReject_LS010 | 0.7117 | 0.7050 | 0.7823 | 0.2123 | 0.7247 | 0.4238 | 0.0000 | 0.6000 |
| SAS-Cert-SoftAR-LS-v1.1 | 0.7102 | 0.7047 | 0.7819 | 0.2105 | 0.7275 | 0.4246 | 0.1000 | 0.5351 |

## Required Answers

1. v1.1 vs Naive: delta BAcc `-0.005404`, delta Macro-F1 `-0.005133`, delta ECE `0.001209`, delta NLL `0.006787`, delta Brier `0.002068`.
2. Better than ArtifactReject on Macro-F1/BAcc: `True`.
3. Better than SoftWeight_noReject on Macro-F1/BAcc: `False`.
4. pyRiemann / MNE-Features status: `used` / `used`.
5. Autoreject status: `available_offline_not_integrated`.
6. Target test leakage: `not detected`; target test is final-evaluation only.
7. Next step: migrate to CBraMod only if v1.1 meets reliability/calibration gates; otherwise revise or park the v1.1 score.

## Output Files

- Metrics: `steegformer_physionetmi_sascert_metrics_v1_1_source_tuned_full.csv`
- Paired comparison: `steegformer_physionetmi_paired_comparison_v1_1_source_tuned_full.csv`
- Score distribution: `certificate_score_distribution_v1_1_source_tuned_full.csv`
- Rejected summary: `rejected_samples_summary_v1_1_source_tuned_full.csv`
- Failure cases: `failure_cases_summary_v1_1_source_tuned_full.csv`
- Leakage audit: `leakage_audit_v1_1_source_tuned_full.json`
- Compact v1.1 result: `compact_sascert_v1_1_result_v1_1_source_tuned_full.json`
