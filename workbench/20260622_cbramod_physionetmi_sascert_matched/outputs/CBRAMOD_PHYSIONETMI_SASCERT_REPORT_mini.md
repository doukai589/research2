# CBraMod PhysioNetMI SAS-Cert Mini Matrix Report

- Targets: `[90, 91, 92]`
- Seeds: `[20, 21]`
- Output tag: `mini`
- Artifact reject percentile: `90.0`
- Raw data copied: `false`
- Raw augmented arrays saved: `false`

## Group Means

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.5349 | 0.4983 | 0.0843 | 0.6931 |
| ArtifactReject_LS010 | 0.5302 | 0.5001 | 0.0923 | 0.6931 |
| SoftWeight_noReject_LS010 | 0.5360 | 0.4925 | 0.0866 | 0.6937 |
| SASCert_SoftAR_LS010 | 0.5473 | 0.4988 | 0.1120 | 0.6932 |

## SASCert vs Naive

- `delta_balanced_accuracy`: `0.012371`
- `delta_macro_f1`: `0.000456`
- `delta_ece`: `0.027748`
- `delta_nll`: `0.000181`
- `delta_brier`: `0.000171`
- `subject_win_rate_macro_f1`: `0.333333`
- `seed_win_rate_macro_f1`: `0.500000`
