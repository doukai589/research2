# SAS-Cert-TFConsistency v0 Report

## Status

- Task: `SASCERT_TFCONSISTENCY_V0_ON_STEEGFORMER_PHYSIONETMI`
- Status: `completed`
- Blocked: `false`
- Decision: `CONTINUE_TFCONSISTENCY_REPAIR`
- Leakage audit: `passed`

## One-Line Conclusion

TFConsistency v0 is directionally healthier than the old weighted-CE branch for calibration, and it slightly improves risk-mixed BAcc, but it does not yet deliver a stable classification gain. It does not meet the risk-mixed success criterion because Macro-F1 gain is near zero and subject/seed win rates are too low.

## Training Card

- Backbone: `ST-EEGFormer-small`
- Checkpoint: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Dataset: `PhysioNetMI / EEGMMI`
- Task: left-vs-right motor imagery
- Runs: `R04/R08/R12`
- Targets: `90-109`
- Seeds: `20-24`
- Frozen modules: ST-EEGFormer-small backbone
- Trainable modules: classifier head only
- Label smoothing: `0.10`
- CertAdapter: not used
- CBraMod / MIRepNet / EEGPT: not used
- Target test use: final evaluation only

## External Augmentation Audit

- DWTaug was reviewed but not used because `pywt` is unavailable in the current environment.
- HHTAug was treated as audit-only because the uploaded source is a script fragment and would need extra integration work.
- Main v0 experiment therefore used controlled TF-style views implemented inside the existing ST runner:
  - `weak_frequency_mask`
  - `weak_time_shift`
  - `weak_amplitude_scaling`
  - `same_class_frequency_mixup`
  - `frequency_mixup`
  - risk-mixed only: `strong_frequency_mask`, `wrong_class_frequency_mixup`, `emg_like_burst`, `eog_like_drift`

## Route Rule

The certificate no longer becomes a scalar CE weight. It routes each augmented view into a training role:

- supervised:
  - `content_q >= 0.67`
  - `risk_q <= 0.50`
- consistency:
  - `content_q >= 0.50`
  - `risk_q <= 0.85`
  - not supervised
- quarantine:
  - all remaining views
  - also NaN/Inf/extreme artifact cases

Loss:

- `L_real = CE(clean, y, label_smoothing=0.10)`
- `L_supervised = CE(supervised_view, y, label_smoothing=0.10)`
- `L_consistency = KL(stopgrad(p_clean) || p_consistency_view)`
- `L = L_real + 1.0 * L_supervised + 2.0 * L_consistency`

Prototype loss is diagnostic-only in v0 because the backbone is frozen and only the classifier head is trainable.

## Regular TF Pool

Main groups:

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| RealOnly_LS010 | 0.752798 | 0.747467 | 0.822920 | 0.188596 | 0.655366 | 0.367650 |
| NaiveTF-Aug_LS010 | 0.754772 | 0.749903 | 0.822609 | 0.183904 | 0.628329 | 0.361606 |
| AugMixTF_LS010 | 0.747479 | 0.742208 | 0.820510 | 0.180920 | 0.595989 | 0.357912 |
| SAS-Cert-TFConsistency_v0 | 0.755226 | 0.749520 | 0.835616 | 0.181557 | 0.602239 | 0.356073 |

SAS-Cert-TFConsistency v0 vs `NaiveTF-Aug_LS010`:

- delta BAcc: `+0.000454`
- delta Macro-F1: `-0.000382`
- delta ECE: `-0.002347`
- delta NLL: `-0.026090`
- delta Brier: `-0.005533`
- subject win rate Macro-F1: `0.45`
- seed win rate Macro-F1: `0.40`

Interpretation:

- Regular pool classification is essentially tied with NaiveTF.
- Calibration improves clearly: NLL and Brier are lower, ECE is slightly lower.
- Macro-F1 is not improved, and win rates are below 0.50, so this is not a stable classification win.

## Risk-Mixed TF Pool

Main groups:

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| RiskMixed_NaiveTF-Aug_LS010 | 0.752664 | 0.747957 | 0.821538 | 0.184812 | 0.624591 | 0.361341 |
| RiskMixed_AugMixTF_LS010 | 0.745454 | 0.739188 | 0.816925 | 0.179009 | 0.602811 | 0.358885 |
| RiskMixed_SAS-Cert-TFConsistency_v0 | 0.755325 | 0.748152 | 0.834232 | 0.183467 | 0.592698 | 0.356317 |

SAS-Cert-TFConsistency v0 vs `RiskMixed_NaiveTF-Aug_LS010`:

- delta BAcc: `+0.002660`
- delta Macro-F1: `+0.000194`
- delta ECE: `-0.001345`
- delta NLL: `-0.031893`
- delta Brier: `-0.005024`
- subject win rate Macro-F1: `0.35`
- seed win rate Macro-F1: `0.00`

SAS-Cert-TFConsistency v0 vs `RiskMixed_AugMixTF_LS010`:

- delta BAcc: `+0.009871`
- delta Macro-F1: `+0.008963`
- delta ECE: `+0.004458`
- delta NLL: `-0.010113`
- delta Brier: `-0.002568`
- subject win rate Macro-F1: `0.35`
- seed win rate Macro-F1: `0.20`

Interpretation:

- In the risk-mixed pool, v0 beats AugMixTF and slightly beats NaiveTF on average.
- However, the success criterion required `delta Macro-F1 or BAcc >= +0.005` versus RiskMixed Naive and `subject win rate >= 0.50`.
- BAcc gain is only `+0.002660`, Macro-F1 gain is only `+0.000194`, subject win rate is `0.35`, and seed win rate is `0.00`.
- Therefore this is not enough to claim reliable augmentation utilization.

## Route Summary

Regular pool route ratios:

| Route | Ratio | Mean content_q | Mean risk_q | Mean raw CE | Mean correctness |
|---|---:|---:|---:|---:|---:|
| supervised | 0.1445 | 0.8368 | 0.2559 | 0.0029 | 1.0000 |
| consistency | 0.2433 | 0.7052 | 0.5390 | 0.0630 | 0.9801 |
| quarantine | 0.6122 | 0.3390 | 0.5421 | 1.8388 | 0.6616 |

Risk-mixed pool route ratios:

| Route | Ratio | Mean content_q | Mean risk_q | Mean raw CE | Mean correctness |
|---|---:|---:|---:|---:|---:|
| supervised | 0.1299 | 0.8382 | 0.2463 | 0.0052 | 0.9992 |
| consistency | 0.2245 | 0.7045 | 0.5368 | 0.0636 | 0.9782 |
| quarantine | 0.6456 | 0.3608 | 0.5382 | 1.8417 | 0.6726 |

Interpretation:

- The route is not completely collapsed: supervised and consistency routes both have non-trivial mass.
- The route is conservative: about 61-65% of candidates are quarantined.
- The route ordering is meaningful diagnostically: supervised views have almost zero raw CE and near-perfect correctness; quarantine views have much higher raw CE and much lower correctness.
- The remaining problem is not that the certificate is useless. The problem is that the fixed route thresholds do not convert the diagnostic separation into a reliable Macro-F1 gain.

## Diagnostic AUC

| Diagnostic | AUC |
|---|---:|
| riskmixed_detection_artifact_score | 0.608842 |
| riskmixed_detection_risk_q | 0.584211 |
| badcontent_wrong_class_frequency_mixup | 0.550457 |
| badartifact_emg_eog | 0.955668 |

Interpretation:

- Artifact diagnostics strongly detect EMG/EOG-style corruption.
- The combined risk score only weakly separates all risk-mixed augmentations from regular ones.
- Wrong-class frequency mixup is not well detected by the current content certificate.
- This explains the mixed training result: the certificate catches obvious artifact risk, but it is not yet strong enough to route all harmful TF views.

## Success Criteria Check

- Regular pool:
  - No major harm versus NaiveTF.
  - Calibration improves.
  - Macro-F1 does not improve.
  - Subject/seed stability is insufficient.
- Risk-mixed pool:
  - BAcc improves by `+0.002660`, below the `+0.005` threshold.
  - Macro-F1 improves by only `+0.000194`.
  - Subject win rate is `0.35`, below `0.50`.
  - Seed win rate is `0.00`.

Final decision:

- `CONTINUE_TFCONSISTENCY_REPAIR`
- Do not enter CBraMod yet.
- Do not claim `SASCERT_USEFUL_WHEN_AUGMENTATION_RISK_EXISTS` yet.

## Recommended Next Step

The next repair should focus on the route/certificate interface, not on adding more tools.

Most useful next tests:

- improve wrong-class / content-risk detection for `wrong_class_frequency_mixup`;
- reduce over-quarantine while preserving the clean supervised route;
- test a content-margin based consistency strength instead of fixed `lambda_cons=2.0`;
- keep artifact/physio/style as diagnostic or route evidence, not direct CE weights.

## Output Files

- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/SASCERT_TFCONSISTENCY_V0_REPORT.md`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/compact_tfconsistency_v0_result.json`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/metrics_regular_tf_pool.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/paired_regular_tf_pool.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/metrics_riskmixed_tf_pool.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/paired_riskmixed_tf_pool.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/route_summary_regular.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/route_summary_riskmixed.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/certificate_distribution_regular.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/certificate_distribution_riskmixed.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/diagnostic_auc_summary.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/augmentation_type_route_table.csv`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/leakage_audit_tfconsistency_v0.json`
- `workbench/20260624_sascert_tfconsistency_v0_steegformer_physionetmi/outputs/failure_review.md`
