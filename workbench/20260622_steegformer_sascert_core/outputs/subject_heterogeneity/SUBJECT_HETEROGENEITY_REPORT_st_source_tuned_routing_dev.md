# Subject Heterogeneity Report

- Output tag: `st_source_tuned_routing_dev`
- Primary: `SoftWeight_noReject_LS010 vs NaiveAug_LS010`
- Secondary: `SASCert_SoftAR_LS010 vs NaiveAug_LS010`
- Subjects: `19`

## Summary

- Primary subject win rate: `0.5263`
- Secondary subject win rate: `0.5263`
- Primary mean subject delta Macro-F1: `0.002870`
- Secondary mean subject delta Macro-F1: `0.003369`
- Decision: `subject_reliability_failed`

## Worst Primary Subjects

| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |
|---:|---:|---:|---:|---:|---:|---:|
| 84 | 0.7329 | -0.0226 | -0.0229 | +0.0170 | 0.2945 | 426.7736 |
| 80 | 0.5851 | -0.0192 | -0.0176 | +0.0051 | 0.2945 | 423.0006 |
| 77 | 0.7362 | -0.0118 | -0.0125 | -0.0038 | 0.2945 | 431.5624 |
| 74 | 0.6330 | -0.0074 | -0.0063 | -0.0083 | 0.2945 | 399.7433 |
| 73 | 0.6413 | -0.0059 | -0.0059 | +0.0132 | 0.2945 | 403.4469 |

## Best Primary Subjects

| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |
|---:|---:|---:|---:|---:|---:|---:|
| 83 | 0.4762 | +0.0329 | +0.0343 | -0.0314 | 0.2945 | 429.3646 |
| 76 | 0.6629 | +0.0273 | +0.0281 | -0.0279 | 0.2945 | 396.1831 |
| 85 | 0.8320 | +0.0176 | +0.0176 | +0.0095 | 0.2945 | 416.6830 |
| 81 | 0.7933 | +0.0173 | +0.0167 | -0.0127 | 0.2945 | 414.7183 |
| 89 | 0.7012 | +0.0157 | +0.0158 | +0.0139 | 0.2945 | 385.6048 |

## Correlations With Primary Delta Macro-F1

- `baseline_macro_f1`: `-0.09042427378189641`
- `baseline_ece`: `-0.02532747427747459`
- `std_content_score`: `0.07128743773175296`
- `mean_artifact_risk`: `-0.19906091973879267`
