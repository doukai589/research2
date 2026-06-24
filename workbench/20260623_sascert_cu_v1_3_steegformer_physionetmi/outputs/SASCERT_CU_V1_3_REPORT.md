# SAS-Cert-CU-LS v1.3 ST-EEGFormer PhysioNetMI Report

## Main Result

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight | Sum weight / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7567 | 0.7524 | 0.8212 | 0.1857 | 0.6221 | 0.3594 | 0.0000 | 1.0000 | 1.0000 |
| SoftWeight_noReject_LS010 | 0.7568 | 0.7527 | 0.8255 | 0.1826 | 0.6150 | 0.3559 | 0.0000 | 0.6000 | 0.6000 |
| SAS-Cert-CU-LS-v1.3 | 0.7588 | 0.7543 | 0.8235 | 0.1861 | 0.6206 | 0.3587 | 0.0000 | 1.0000 | 1.0000 |

## Component Utility Audit

- Training utility candidates: `['E_embed', 'E_proto', 'E_content']`.
- Diagnostic/unstable scores: `['E_pred', 'artifact_score', 'artifact_safe', 'E_physio', 'E_style', 'D_band', 'D_cov', 'D_style']`.

| Score | Spearman CE | Spearman correctness | Top30-bottom30 CE | Top30-bottom30 correctness | Role |
|---|---:|---:|---:|---:|---|
| E_embed | -0.0892 | 0.0503 | -0.3776 | 0.0679 | `training_utility_candidate` |
| E_proto | -0.8877 | 0.4623 | -3.8085 | 0.6917 | `training_utility_candidate` |
| E_pred | -0.7553 | 0.2393 | -0.9929 | 0.3756 | `diagnostic_report_only` |
| E_content | -0.5293 | 0.2972 | -2.4121 | 0.4206 | `training_utility_candidate` |
| artifact_score | 0.0252 | -0.0156 | 0.2217 | -0.0233 | `diagnostic_report_only` |
| artifact_safe | -0.0339 | 0.0177 | -0.1999 | 0.0256 | `diagnostic_report_only` |
| E_physio | -0.0049 | 0.0161 | -0.1088 | 0.0272 | `diagnostic_report_only` |
| E_style | -0.0220 | 0.0212 | -0.1927 | 0.0322 | `diagnostic_report_only` |
| D_band | 0.0040 | -0.0143 | 0.0602 | -0.0278 | `diagnostic_report_only` |
| D_cov | 0.0164 | -0.0666 | 0.2185 | -0.0361 | `diagnostic_report_only` |
| D_style | 0.0444 | 0.0137 | 0.0703 | -0.0383 | `diagnostic_report_only` |

## Required Answers

- v1.3 vs Naive: delta BAcc `0.002078`, delta Macro-F1 `0.001935`, delta ECE `0.000444`, delta NLL `-0.001490`, delta Brier `-0.000763`.
- v1.3 vs SoftWeight: delta BAcc `0.001959`, delta Macro-F1 `0.001634`, delta ECE `0.003503`, delta NLL `0.005643`, delta Brier `0.002736`.
- Subject win rate Macro-F1 vs Naive: `0.050000`.
- Seed win rate Macro-F1 vs Naive: `0.000000`.
- Mean weight / weight range: `1.000000` / `[0.75, 1.25]`.
- Nonfinite content skipped count: `0`.
- Target test leakage: `not detected`.
- Decision: `enter_cbramod_recheck`.
- Next recommendation: `进入 CBraMod 复验`.

## Output Files

- `SASCERT_CU_V1_3_REPORT.md`
- `compact_sascert_v1_3_result.json`
- `metrics_v1_3.csv`
- `paired_comparison_v1_3.csv`
- `component_utility_audit.csv`
- `component_utility_summary.json`
- `diagnostic_scores_v1_3.csv`
- `leakage_audit_v1_3.json`
