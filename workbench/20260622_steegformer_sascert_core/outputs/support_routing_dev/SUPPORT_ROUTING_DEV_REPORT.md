# Support Routing Dev Report

- Output tag: `st_source_tuned_routing_dev`
- Score tag: `st_source_tuned_routing_dev`
- Dev folds: `95`
- Dev subjects: `19`

## Baselines

| Strategy | Macro-F1 | BAcc | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.6621 | 0.6665 | 0.2350 | 0.8119 | 0.4868 |
| SoftWeight_noReject_LS010 | 0.6649 | 0.6700 | 0.2316 | 0.8091 | 0.4822 |
| SASCert_SoftAR_LS010 | 0.6654 | 0.6700 | 0.2333 | 0.8062 | 0.4813 |
| LOSO support routing | 0.6594 | 0.6641 | 0.2327 | 0.8132 | 0.4852 |

## Full-Dev Best Rule

- Rule: `threshold:mean_artifact_risk<=475.104?SASCert_SoftAR_LS010:NaiveAug_LS010`
- Macro-F1: `0.667522`

## Decision

`do_not_freeze_routing_rule`

LOSO support routing does not beat the best constant dev strategy under the current support-only features. Keep this as diagnostic evidence; do not apply a routed method to final target subjects yet.
