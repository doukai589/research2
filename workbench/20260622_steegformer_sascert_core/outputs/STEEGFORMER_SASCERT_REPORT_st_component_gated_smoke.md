# ST-EEGFormer PhysioNetMI SAS-Cert Workbench Report

- Targets: `[90]`
- Seeds: `[20]`
- Smoke: `True`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `st_component_gated_smoke`
- Experiment: `component_gated`
- Groups: `['NaiveAug_LS010', 'SoftWeight_noReject_LS010', 'SASCert_SoftAR_LS010', 'ArtifactGatePhysio_LS010', 'ComponentGatedV1_LS010']`
- Primary group: `ComponentGatedV1_LS010`
- Artifact reject percentile: `90.0`

## Primary vs Naive

- `primary_group`: `ComponentGatedV1_LS010`
- `baseline_group`: `NaiveAug_LS010`
- `delta_balanced_accuracy`: `0.000000`
- `delta_macro_f1`: `0.000000`
- `delta_ece`: `0.067403`
- `delta_nll`: `0.048643`
- `delta_brier`: `0.026557`
- `subject_win_rate_macro_f1`: `0.000000`
- `seed_win_rate_macro_f1`: `0.000000`
