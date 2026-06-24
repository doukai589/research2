# SAS-Cert v2 Structured Certificate Adapter Report

- Status: `completed`
- Decision: `continue_repair_v2_or_diagnostic_only`
- Leakage audit: `passed`

## Training Card

- Backbone: `ST-EEGFormer-small source-tuned checkpoint`
- Frozen modules: `ST-EEGFormer-small backbone`
- Trainable modules: `Task-Prior Head`, `classifier head`, `CertAdapter for v2-full`
- Target test: `final evaluation only`

## Regular Pool

- v2-full vs Naive: `{'delta_balanced_accuracy': 0.00039140018907812824, 'delta_macro_f1': -0.001637930078380756, 'delta_ece': -0.0014698388012096275, 'delta_nll': -0.008335580438585843, 'delta_brier': -0.00017790588550270492}`
- v2-full vs CU-v1.3: `{'delta_balanced_accuracy': -0.0016868258074915765, 'delta_macro_f1': -0.0035724317136389683, 'delta_ece': -0.001913467552495024, 'delta_nll': -0.006845284779359417, 'delta_brier': 0.0005851526185870481}`
- v2-full vs no-adapter: `{'delta_balanced_accuracy': -0.0016429992951666472, 'delta_macro_f1': -0.003464703891687293, 'delta_ece': 0.001723010135261882, 'delta_nll': 0.01145286260454792, 'delta_brier': 0.004032667167484794}`
- v2-no-adapter vs CU-v1.3: `{'delta_balanced_accuracy': -4.382651232492929e-05, 'delta_macro_f1': -0.0001077278219516753, 'delta_ece': -0.003636477687756906, 'delta_nll': -0.018298147383907337, 'delta_brier': -0.0034475145488977454}`
- win rates vs Naive: `{'subject_win_rate_macro_f1': 0.2, 'seed_win_rate_macro_f1': 0.2}`

## Risk-Mixed Pool

- v2-full vs RiskMixed Naive: `{'delta_balanced_accuracy': -0.0007264342339806662, 'delta_macro_f1': -0.0016512914640610665, 'delta_ece': 0.0022635599372663184, 'delta_nll': -0.01183946294268201, 'delta_brier': -0.0028650696668773845}`
- v2-full vs RiskMixed CU-v1.3: `{'delta_balanced_accuracy': -0.0008025240798074096, 'delta_macro_f1': -0.001616944462463632, 'delta_ece': 0.001832344337337144, 'delta_nll': -0.007981320597778119, 'delta_brier': -0.0005322832241654307}`
- win rates: `{'subject_win_rate_macro_f1': 0.25, 'seed_win_rate_macro_f1': 0.0}`

## Certificate Summary

- Gamma / agreement: `{'gamma': [{'pool': 'regular', 'aug_type': 'ALL', 'n': 6000, 'gamma_mean': 0.6528052739767979, 'gamma_std': 0.3076037050576863, 'gamma_p10': 0.05394231155514717, 'gamma_p50': 0.8292162120342255, 'gamma_p90': 0.861110943555832, 'artifact_score_mean': 1.581825144531826, 'E_physio_mean': 0.4999999995343387, 'E_style_mean': 0.4999999995343387}, {'pool': 'riskmixed', 'aug_type': 'ALL', 'n': 10000, 'gamma_mean': 0.6532956597868353, 'gamma_std': 0.30706944732140207, 'gamma_p10': 0.053662870079278946, 'gamma_p50': 0.8289620876312256, 'gamma_p90': 0.861056387424469, 'artifact_score_mean': 3.9535120114721356, 'E_physio_mean': 0.5000000001024455, 'E_style_mean': 0.5000000001024455}], 'prototype_agreement': [{'pool': 'regular', 'aug_type': 'ALL', 'n': 6000, 'prior_agreement_rate': 0.7825, 'prototype_agreement_rate': 0.7785, 'both_agreement_rate': 0.7381666666666666, 'proto_margin_mean': 0.03095110168059667, 'prior_margin_mean': 2.555417570869128}, {'pool': 'riskmixed', 'aug_type': 'ALL', 'n': 10000, 'prior_agreement_rate': 0.7864, 'prototype_agreement_rate': 0.7767, 'both_agreement_rate': 0.7401, 'proto_margin_mean': 0.030524500328302385, 'prior_margin_mean': 2.559306250834465}]}`

## Output Files

- `TRAINING_PLAN.md`
- `TRAINING_REPORT.md`
- `SASCERT_V2_REPORT.md`
- `compact_sascert_v2_result.json`
- `metrics_regular_pool.csv`
- `paired_regular_pool.csv`
- `metrics_riskmixed_pool.csv`
- `paired_riskmixed_pool.csv`
- `certificate_gamma_distribution.csv`
- `prototype_agreement_summary.csv`
- `adapter_ablation_summary.csv`
- `leakage_audit_v2.json`
