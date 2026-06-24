# Subject Heterogeneity Report

- Output tag: `st_source_tuned_full`
- Primary: `SoftWeight_noReject_LS010 vs NaiveAug_LS010`
- Secondary: `SASCert_SoftAR_LS010 vs NaiveAug_LS010`
- Subjects: `20`

## Summary

- Primary subject win rate: `0.6000`
- Secondary subject win rate: `0.6500`
- Primary mean subject delta Macro-F1: `0.006402`
- Secondary mean subject delta Macro-F1: `0.006289`
- Decision: `subject_reliability_failed`

## Worst Primary Subjects

| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |
|---:|---:|---:|---:|---:|---:|---:|
| 105 | 0.9244 | -0.0174 | -0.0176 | -0.0028 | 0.2945 | 418.0174 |
| 93 | 0.5530 | -0.0135 | -0.0170 | +0.0170 | 0.2945 | 416.6027 |
| 102 | 0.6411 | -0.0119 | -0.0111 | +0.0404 | 0.2945 | 393.5102 |
| 101 | 0.9714 | -0.0057 | -0.0056 | +0.0043 | 0.2945 | 426.6153 |
| 97 | 0.7294 | -0.0051 | -0.0052 | -0.0027 | 0.2945 | 427.5371 |

## Best Primary Subjects

| Subject | Baseline Macro-F1 | Delta Macro-F1 | Delta BAcc | Delta ECE | Content Std | Artifact Risk |
|---:|---:|---:|---:|---:|---:|---:|
| 106 | 0.5623 | +0.0408 | +0.0418 | -0.0317 | 0.2945 | 428.0146 |
| 108 | 0.6642 | +0.0397 | +0.0392 | -0.0268 | 0.2945 | 429.0743 |
| 98 | 0.5135 | +0.0258 | +0.0284 | -0.0527 | 0.2945 | 403.6761 |
| 95 | 0.6565 | +0.0147 | +0.0186 | +0.0257 | 0.2945 | 433.2610 |
| 94 | 0.6717 | +0.0146 | +0.0118 | -0.0168 | 0.2945 | 429.7091 |

## Correlations With Primary Delta Macro-F1

- `baseline_macro_f1`: `-0.42088409387404135`
- `baseline_ece`: `0.35250655477361403`
- `std_content_score`: `0.01816757991891089`
- `mean_artifact_risk`: `0.2270833751264425`
