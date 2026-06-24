# ST-EEGFormer PhysioNetMI SAS-Cert v1.1 Workbench Report

- Targets: `[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]`
- Seeds: `[20, 21, 22, 23, 24]`
- Smoke: `False`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `v1_3`
- Experiment: `v1_3`
- Groups: `['NaiveAug_LS010', 'SoftWeight_noReject_LS010', 'SAS-Cert-CU-LS-v1.3']`
- Primary group: `SAS-Cert-CU-LS-v1.3`
- Artifact reject percentile: `90.0`

## Primary vs Naive

- `primary_group`: `SAS-Cert-CU-LS-v1.3`
- `baseline_group`: `NaiveAug_LS010`
- `delta_balanced_accuracy`: `0.002078`
- `delta_macro_f1`: `0.001935`
- `delta_ece`: `0.000444`
- `delta_nll`: `-0.001490`
- `delta_brier`: `-0.000763`
- `subject_win_rate_macro_f1`: `0.050000`
- `seed_win_rate_macro_f1`: `0.000000`
