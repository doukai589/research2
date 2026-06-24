# SAS-Cert-TFConsistency v0 Training Report

## Training Card Summary

- Backbone: `ST-EEGFormer-small`
- Checkpoint: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Dataset/task: `PhysioNetMI / EEGMMI`, left-vs-right motor imagery, runs `R04/R08/R12`
- Targets/seeds: targets `90-109`, seeds `20-24`
- Frozen: ST backbone
- Trainable: classifier head only
- Loss: `CE(real) + 1.0 * CE(supervised views) + 2.0 * KL(consistency views)`
- Label smoothing: `0.10`
- Prototype loss: diagnostic-only in v0
- Target test: final evaluation only

## Route Implementation

- `content_q = ranknorm(E_embed + E_proto)` computed inside the support candidate pool.
- `risk_q = ranknorm(artifact_score + physio_deviation)` computed inside the support candidate pool.
- supervised route: `content_q >= 0.67` and `risk_q <= 0.50`.
- consistency route: `content_q >= 0.50` and `risk_q <= 0.85`, excluding supervised views.
- quarantine route: all remaining, NaN/Inf, or extreme artifact views.

## Main Result

Regular TF pool:

- v0 vs NaiveTF:
  - delta BAcc `+0.000454`
  - delta Macro-F1 `-0.000382`
  - delta ECE `-0.002347`
  - delta NLL `-0.026090`
  - delta Brier `-0.005533`
  - subject win rate `0.45`
  - seed win rate `0.40`

Risk-mixed TF pool:

- v0 vs RiskMixed NaiveTF:
  - delta BAcc `+0.002660`
  - delta Macro-F1 `+0.000194`
  - delta ECE `-0.001345`
  - delta NLL `-0.031893`
  - delta Brier `-0.005024`
  - subject win rate `0.35`
  - seed win rate `0.00`

## Interpretation

The route is diagnostically meaningful: supervised views have very low raw CE and almost perfect correctness, consistency views are mostly correct, and quarantined views have much higher CE. But the route is too conservative and not reliable enough for classification gain. About 61-65% of candidates are quarantined, while the risk-mixed improvement over NaiveTF is below the required `+0.005` BAcc/Macro-F1 threshold.

The strongest diagnostic success is artifact detection for EMG/EOG-like corruption (`AUC = 0.955668`). The weakest important diagnostic is wrong-class frequency mixup detection (`AUC = 0.550457`), which means the current content certificate still misses a key harmful augmentation family.

## Decision

- Status: `completed`
- Blocked: `false`
- Leakage: `passed`
- Decision: `CONTINUE_TFCONSISTENCY_REPAIR`
- Do not enter CBraMod yet.
- Do not claim reliable risk-mixed augmentation utilization yet.
