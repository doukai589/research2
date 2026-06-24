# CBraMod PhysioNetMI SAS-Cert Mini Matrix Report

- Targets: `[90]`
- Seeds: `[20]`
- Output tag: `repaired_mini_smoke`
- Experiment: `repaired`
- Artifact reject percentile: `90.0`
- Raw data copied: `false`
- Raw augmented arrays saved: `false`

## Group Means

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.3676 | 0.3583 | 0.1318 | 0.6951 |
| NoAug_LS010 | 0.3399 | 0.3342 | 0.1601 | 0.6950 |
| RepairedSoftWeight_artifact_gate_physio_LS010 | 0.3660 | 0.3452 | 0.1318 | 0.6951 |

## Primary vs Naive

- `delta_balanced_accuracy`: `-0.001634`
- `delta_macro_f1`: `-0.013095`
- `delta_ece`: `0.000070`
- `delta_nll`: `-0.000065`
- `delta_brier`: `-0.000065`
- `subject_win_rate_macro_f1`: `0.000000`
- `seed_win_rate_macro_f1`: `0.000000`
