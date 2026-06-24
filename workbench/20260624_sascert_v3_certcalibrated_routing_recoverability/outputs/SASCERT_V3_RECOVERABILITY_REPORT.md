# SAS-Cert v3 Certificate-Calibrated Routing Recoverability Test

- Status: `completed`
- Decision: `NO_RECOVERABLE_TRAINING_GAIN`
- Leakage audit: `passed`

## Training Card

- Backbone: `ST-EEGFormer-small source-tuned checkpoint`
- Frozen modules: `ST-EEGFormer-small backbone`
- Trainable modules: `classifier head only for oracle routing`
- Oracle risk label: risky augmentation type list; target test not used
- Loss: `CE(real) + normalized CE(mild augmented candidates)`
- Baseline: `RiskMixed_NaiveAug_LS010`

## Oracle Risk Routing

- OracleRiskReject vs RiskMixed Naive: `{'delta_balanced_accuracy': 0.0020839392375771038, 'delta_macro_f1': 0.0019210518538772536, 'delta_ece': 1.0344514208004973e-06, 'delta_nll': 0.006000249215977971, 'delta_brier': -0.0006380253378301948}`
- Win rates: `{'subject_win_rate_macro_f1': 0.1, 'seed_win_rate_macro_f1': 0.0}`
- Routing summary: `[{'route': 'supervised_ce_mild', 'n': 7000, 'ratio': 0.7, 'mean_artifact_score': 1.7174578081699354}, {'route': 'quarantine_risky', 'n': 3000, 'ratio': 0.3, 'mean_artifact_score': 9.17097181917727}]`

## Calibrator

- Status: `skipped`
- Reason: `Oracle failed recoverability test.`

## Output Files

- `SASCERT_V3_RECOVERABILITY_REPORT.md`
- `compact_sascert_v3_result.json`
- `oracle_routing_metrics.csv`
- `calibrator_validation_metrics.csv`
- `certcalibrated_routing_metrics.csv`
- `paired_comparison_v3.csv`
- `p_bad_distribution.csv`
- `routing_summary.csv`
- `leakage_audit_v3.json`
- `failure_review.md` if failed
