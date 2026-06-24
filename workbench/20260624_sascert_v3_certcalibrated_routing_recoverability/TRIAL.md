# sascert_v3_certcalibrated_routing_recoverability

## Intent

Recoverability test after v2: determine whether oracle risky-augmentation
routing can recover classification gain before training any certificate
calibrator.

## Protocol

- Backbone: ST-EEGFormer-small source-tuned checkpoint.
- Dataset: PhysioNetMI / EEGMMI left-vs-right motor imagery.
- Runs: R04/R08/R12.
- Targets: 90-109.
- Seeds: 20-24.
- ST backbone frozen.
- No CertAdapter.

## Oracle Risk Routing

Risky augmentation types:

- `strong_frequency_mask`
- `strong_channel_dropout`
- `emg_like_burst`
- `eog_like_drift`
- `covariance_perturbation`

Route:

- mild augmentation: supervised CE
- risky augmentation: quarantine from supervised CE

## Decision

- `NO_RECOVERABLE_TRAINING_GAIN`
- OracleRiskReject did not meet the recoverability success criterion.
- Certificate calibrator full training was skipped by protocol.

## Results

OracleRiskReject vs `RiskMixed_NaiveAug_LS010`:

- delta BAcc `+0.002084`
- delta Macro-F1 `+0.001921`
- delta ECE `+0.000001`
- delta NLL `+0.006000`
- delta Brier `-0.000638`
- subject / seed win rate Macro-F1: `0.10 / 0.00`

Routing:

- supervised mild candidates:
  - n `7000`
  - ratio `0.70`
  - mean artifact score `1.7175`
- quarantined risky candidates:
  - n `3000`
  - ratio `0.30`
  - mean artifact score `9.1710`

Leakage audit:

- `passed`
- target test was used only for final evaluation.
