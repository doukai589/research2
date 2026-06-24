# sascert_v2_certadapter_steegformer_physionetmi

## Intent

Implement and validate SAS-Cert v2 as structured augmentation-certificate
guided adaptation, rather than another v1.x heuristic weight repair.

## Protocol

- Backbone: ST-EEGFormer-small.
- Dataset: PhysioNetMI / EEGMMI left-vs-right motor imagery.
- Runs: R04/R08/R12.
- Targets: 90-109.
- Seeds: 20-24.
- Source-tuned checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`.
- ST backbone remains frozen.

## Groups

Regular pool:

- `NaiveAug_LS010`
- `SAS-Cert-CU-LS-v1.3`
- `SAS-Cert-v2-no-adapter`
- `SAS-Cert-v2-full`

Risk-mixed pool:

- `RiskMixed_NaiveAug_LS010`
- `RiskMixed_CU-v1.3`
- `RiskMixed_SAS-Cert-v2-full`

## Decision

- `continue_repair_v2_or_diagnostic_only`
- v2-full did not satisfy the regular-pool or risk-mixed success criteria.
- CertAdapter underperformed v2-no-adapter on regular-pool Macro-F1, so the
  current adapter form should not be promoted or stacked with more modules.

## Results

Regular pool, v2-full vs `NaiveAug_LS010`:

- delta BAcc `+0.000391`
- delta Macro-F1 `-0.001638`
- delta ECE `-0.001470`
- delta NLL `-0.008336`
- delta Brier `-0.000178`
- subject / seed win rate Macro-F1: `0.20 / 0.20`

Regular pool ablation:

- v2-full vs CU-v1.3:
  - delta BAcc `-0.001687`
  - delta Macro-F1 `-0.003572`
- v2-full vs v2-no-adapter:
  - delta BAcc `-0.001643`
  - delta Macro-F1 `-0.003465`
- v2-no-adapter vs CU-v1.3:
  - delta BAcc `-0.000044`
  - delta Macro-F1 `-0.000108`

Risk-mixed pool, v2-full vs `RiskMixed_NaiveAug_LS010`:

- delta BAcc `-0.000726`
- delta Macro-F1 `-0.001651`
- delta ECE `+0.002264`
- delta NLL `-0.011839`
- delta Brier `-0.002865`
- subject / seed win rate Macro-F1: `0.25 / 0.00`

Certificate summary:

- regular gamma mean `0.6528`, p10/p50/p90 `0.0539 / 0.8292 / 0.8611`
- risk-mixed gamma mean `0.6533`, p10/p50/p90 `0.0537 / 0.8290 / 0.8611`
- regular prior/prototype/both agreement:
  - `0.7825 / 0.7785 / 0.7382`
- risk-mixed prior/prototype/both agreement:
  - `0.7864 / 0.7767 / 0.7401`

Leakage audit:

- `passed`
- target test was used only for final evaluation.
