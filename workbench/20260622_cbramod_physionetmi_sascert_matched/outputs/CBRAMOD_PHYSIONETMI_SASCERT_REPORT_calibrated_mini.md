# CBraMod PhysioNetMI SAS-Cert Mini Matrix Report

- Targets: `[90, 91, 92]`
- Seeds: `[20, 21]`
- Output tag: `calibrated_mini`
- Experiment: `calibrated`
- Artifact reject percentile: `90.0`
- Raw data copied: `false`
- Raw augmented arrays saved: `false`

## Group Means

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.5349 | 0.4983 | 0.0843 | 0.6931 |
| NoAug_LS010 | 0.5231 | 0.4954 | 0.0595 | 0.6927 |
| RepairedSoftWeightTemp_artifact_gate_physio_LS010 | 0.5461 | 0.5409 | 0.1064 | 0.6940 |
| RepairedSoftWeight_artifact_gate_physio_LS010 | 0.5461 | 0.5409 | 0.1070 | 0.6916 |

## Primary vs Naive

- `delta_balanced_accuracy`: `0.011124`
- `delta_macro_f1`: `0.042646`
- `delta_ece`: `0.022131`
- `delta_nll`: `0.000938`
- `delta_brier`: `0.000654`
- `subject_win_rate_macro_f1`: `0.666667`
- `seed_win_rate_macro_f1`: `1.000000`
