# CBraMod PhysioNetMI SAS-Cert Mini Matrix Report

- Targets: `[90]`
- Seeds: `[20]`
- Output tag: `mini_smoke`
- Artifact reject percentile: `90.0`
- Raw data copied: `false`
- Raw augmented arrays saved: `false`

## Group Means

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.3676 | 0.3583 | 0.1318 | 0.6951 |
| ArtifactReject_LS010 | 0.4330 | 0.4167 | 0.0748 | 0.6953 |
| SoftWeight_noReject_LS010 | 0.4722 | 0.3269 | 0.0198 | 0.6951 |
| SASCert_SoftAR_LS010 | 0.3889 | 0.2857 | 0.1047 | 0.6951 |

## SASCert vs Naive

- `delta_balanced_accuracy`: `0.021242`
- `delta_macro_f1`: `-0.072619`
- `delta_ece`: `-0.027104`
- `delta_nll`: `-0.000026`
- `delta_brier`: `-0.000026`
- `subject_win_rate_macro_f1`: `0.000000`
- `seed_win_rate_macro_f1`: `0.000000`
