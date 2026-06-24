# sascert_tfconsistency_v0_steegformer_physionetmi

## Intent

Test SAS-Cert-TFConsistency v0: route EEG time-frequency augmentation views
into supervised, consistency, or quarantine training roles.

## Protocol

- Backbone: ST-EEGFormer-small source-tuned checkpoint.
- Dataset: PhysioNetMI / EEGMMI left-vs-right motor imagery.
- Runs: R04/R08/R12.
- Targets: 90-109.
- Seeds: 20-24.
- ST backbone frozen.
- Trainable module: classifier head only.

## Groups

Regular TF pool:

- `RealOnly_LS010`
- `NaiveTF-Aug_LS010`
- `AugMixTF_LS010`
- `SAS-Cert-TFConsistency_v0`

Risk-mixed TF pool:

- `RiskMixed_NaiveTF-Aug_LS010`
- `RiskMixed_AugMixTF_LS010`
- `RiskMixed_SAS-Cert-TFConsistency_v0`

## Decision

- completed

TFConsistency v0 did not meet the risk-mixed success criterion. It improved
calibration and slightly improved risk-mixed BAcc, but Macro-F1 gain was near
zero and subject/seed win rates were too low.

- Decision: `CONTINUE_TFCONSISTENCY_REPAIR`
- Regular v0 vs NaiveTF:
  - delta BAcc: `+0.000454`
  - delta Macro-F1: `-0.000382`
- Risk-mixed v0 vs RiskMixed NaiveTF:
  - delta BAcc: `+0.002660`
  - delta Macro-F1: `+0.000194`
  - subject win rate: `0.35`
  - seed win rate: `0.00`
