# SAS-Cert-SoftAR-LS v1.1 ST-EEGFormer PhysioNetMI Report

## Scope

- Targets: `[90]`
- Seeds: `[20]`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `v1_1_smoke`
- Source-tuned checkpoint: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Candidate augmentations per support trial: `6`
- Artifact reject percentile: `90.0`

## Main Result

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.9722 | 0.9714 | 0.9967 | 0.0315 | 0.0707 | 0.0438 | 0.0000 | 1.0000 |
| ArtifactReject_LS010 | 0.8056 | 0.7939 | 0.9967 | 0.1351 | 0.4711 | 0.2779 | 0.1000 | 1.0000 |
| SoftWeight_noReject_LS010 | 0.9722 | 0.9714 | 0.9967 | 0.0452 | 0.0827 | 0.0520 | 0.0000 | 0.6000 |
| SAS-Cert-SoftAR-LS-v1.1 | 0.8333 | 0.8250 | 0.9967 | 0.0938 | 0.4220 | 0.2563 | 0.1000 | 0.5418 |

## Required Answers

1. v1.1 vs Naive: delta BAcc `-0.138889`, delta Macro-F1 `-0.146429`, delta ECE `0.062346`, delta NLL `0.351290`, delta Brier `0.212491`.
2. Better than ArtifactReject on Macro-F1/BAcc: `True`.
3. Better than SoftWeight_noReject on Macro-F1/BAcc: `False`.
4. pyRiemann / MNE-Features status: `used` / `used`.
5. Autoreject status: `available_offline_not_integrated`.
6. Target test leakage: `not detected`; target test is final-evaluation only.
7. Next step: migrate to CBraMod only if v1.1 meets reliability/calibration gates; otherwise revise or park the v1.1 score.

## Output Files

- Metrics: `steegformer_physionetmi_sascert_metrics_v1_1_smoke.csv`
- Paired comparison: `steegformer_physionetmi_paired_comparison_v1_1_smoke.csv`
- Score distribution: `certificate_score_distribution_v1_1_smoke.csv`
- Rejected summary: `rejected_samples_summary_v1_1_smoke.csv`
- Failure cases: `failure_cases_summary_v1_1_smoke.csv`
- Leakage audit: `leakage_audit_v1_1_smoke.json`
- Compact v1.1 result: `compact_sascert_v1_1_result_v1_1_smoke.json`
