# ST-EEGFormer PhysioNetMI SAS-Cert Workbench Report

- Targets: `[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]`
- Seeds: `[20, 21, 22, 23, 24]`
- Smoke: `False`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `st_source_tuned_routing_dev`
- Artifact reject percentile: `90.0`

## SASCert vs Naive

- `delta_balanced_accuracy`: `0.003514`
- `delta_macro_f1`: `0.003369`
- `delta_ece`: `-0.001708`
- `delta_nll`: `-0.005634`
- `delta_brier`: `-0.005467`
- `subject_win_rate_macro_f1`: `0.263158`
- `seed_win_rate_macro_f1`: `0.000000`
