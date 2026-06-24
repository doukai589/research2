# Support/Candidate Utility Alignment Audit

## Scope

- Inputs: existing `st_source_tuned_full` metrics and score rows
- No new model training
- No new dataset or backbone
- Purpose: test whether fold-level candidate/support summaries explain `SoftWeight_noReject_LS010` utility

## Top Correlations With SoftWeight Benefit

| Feature | Spearman | Pearson |
|---|---:|---:|
| `clean_artifact_risk_raw_mean` | 0.1168 | 0.1072 |
| `sas_score_p10` | -0.1108 | -0.1387 |
| `artifact_risk_raw_mean` | 0.1078 | 0.1244 |
| `artifact_risk_raw_p10` | 0.0933 | 0.1172 |
| `artifact_risk_raw_std` | 0.0839 | 0.1144 |
| `clean_content_score_mean` | 0.0622 | 0.1445 |
| `sas_score_p90` | 0.0599 | 0.0141 |
| `artifact_risk_raw_p90` | 0.0449 | 0.1012 |
| `sas_score_mean` | 0.0278 | 0.0088 |
| `sas_score_std` | 0.0129 | -0.0287 |

## Subject-Level Pattern

Best subjects:

| Subject | Mean Delta Macro-F1 | Baseline Macro-F1 | Positive Seed Fraction |
|---:|---:|---:|---:|
| 106 | 0.0408 | 0.5623 | 1.0000 |
| 108 | 0.0397 | 0.6642 | 0.8000 |
| 98 | 0.0258 | 0.5135 | 0.4000 |
| 95 | 0.0147 | 0.6565 | 0.4000 |
| 94 | 0.0146 | 0.6717 | 0.4000 |

Worst subjects:

| Subject | Mean Delta Macro-F1 | Baseline Macro-F1 | Positive Seed Fraction |
|---:|---:|---:|---:|
| 105 | -0.0174 | 0.9244 | 0.2000 |
| 93 | -0.0135 | 0.5530 | 0.2000 |
| 102 | -0.0119 | 0.6411 | 0.2000 |
| 101 | -0.0057 | 0.9714 | 0.0000 |
| 97 | -0.0051 | 0.7294 | 0.2000 |

## Decision

`park_st_weighting_variants`

The existing-output audit did not find a strong enough fold-level alignment signal. Under the stop rule, ST weighting variants should be parked or reframed as diagnostic observations rather than expanded.
