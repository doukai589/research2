# SAS-Cert v1.4 SCB-CU Risk-Mixed Stress Test

- Status: `completed`
- Decision: `limit_training_use_to_diagnostic_or_riskmixed`
- Leakage audit: `passed`

## Regular Pool

- v1.4 vs Naive: `{'delta_balanced_accuracy': 0.0006255735793274653, 'delta_macro_f1': 0.0002947992557784662, 'delta_ece': -0.0014123958752163823, 'delta_nll': -0.0019076668427487897, 'delta_brier': -0.00097735212184491}`
- v1.4 vs v1.3: `{'delta_balanced_accuracy': -0.0014526524172422395, 'delta_macro_f1': -0.001639702379479746, 'delta_ece': -0.0018560246265017788, 'delta_nll': -0.0004173711835223637, 'delta_brier': -0.00021429361775515687}`
- v1.4 vs Naive win rates: `{'subject_win_rate_macro_f1': 0.0, 'seed_win_rate_macro_f1': 0.0}`
- v1.4 vs v1.3 win rates: `{'subject_win_rate_macro_f1': 0.0, 'seed_win_rate_macro_f1': 0.0}`

## Localization Audit

- Summary: `{'weak_proto_subjects': ['101'], 'class_unfair_subjects': [], 'mean_subject_delta_macro_f1': 0.0002947992557783607, 'subject_win_rate_macro_f1_vs_naive': 0.3, 'mean_subject_delta_bacc': 0.0006255735793274375, 'class_balance_min_over_max_mean': 1.0, 'component_corr_mean': {'E_content': {'spearman_raw_CE_loss': -0.5668992325328263, 'spearman_correctness': 0.3305460963113224}, 'E_embed': {'spearman_raw_CE_loss': -0.07480145316705647, 'spearman_correctness': 0.0794151044846301}, 'E_proto': {'spearman_raw_CE_loss': -0.8502974475788093, 'spearman_correctness': 0.4572056376402097}}, 'v1_4_weight_mean_by_subject_class_min': 1.0, 'v1_4_weight_mean_by_subject_class_max': 1.0, 'ranknorm_scopes': ['subject_class']}`

## Risk-Mixed Pool

- v1.4 vs RiskMixed Naive: `{'delta_balanced_accuracy': 0.0004759866466522711, 'delta_macro_f1': 0.00045003224939599296, 'delta_ece': -0.0011207446146252464, 'delta_nll': -0.003801229873548362, 'delta_brier': -0.0022358023282140205}`
- win rates: `{'subject_win_rate_macro_f1': 0.0, 'seed_win_rate_macro_f1': 0.0}`

## Output Files

- `metrics_v1_4_regular_pool.csv`
- `paired_comparison_v1_4_regular_pool.csv`
- `metrics_v1_4_riskmixed_pool.csv`
- `paired_comparison_v1_4_riskmixed_pool.csv`
- `per_subject_delta_table.csv`
- `per_class_delta_table.csv`
- `per_subject_component_corr.csv`
- `weight_distribution_by_subject_class.csv`
- `riskmixed_diagnostic_summary.csv`
- `leakage_audit_v1_4.json`
- `compact_sascert_v1_4_result.json`
