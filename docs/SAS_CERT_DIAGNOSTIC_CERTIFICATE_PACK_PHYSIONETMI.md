# SAS-Cert Diagnostic Certificate Pack: PhysioNetMI

## Scope

- Dataset: `PhysioNetMI`, left/right motor imagery, runs `R04/R08/R12`
- Backbones: `CBraMod_frozen`, `ST-EEGFormer-small_source_tuned`
- Mode: existing-output-only diagnostic MVE
- No new training, no new dataset, no new backbone

## Diagnostic Score Evidence

| Backbone | Current SAS AUC | Component-Gated v1 AUC | Artifact-Gate Physio AUC | Physio AUC | Style AUC | Content AUC | Artifact-Safe AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CBraMod_frozen` | 0.1969 | 0.8395 | 0.9022 | 0.8444 | 0.6408 | 0.3044 | 0.3333 |
| `ST-EEGFormer-small_source_tuned` | 0.1662 | 0.8395 | 0.9022 | 0.8444 | 0.6408 | 0.2722 | 0.3333 |

The old scalar SAS score fails directionally on the mixed-bad PhysioNetMI pool, while component-gated and artifact-gate-physio variants recover strong diagnostic AUC on both backbones.

## Training Policy Evidence

| Backbone | Branch | Scope | Delta BAcc | Delta Macro-F1 | Delta ECE | Decision |
|---|---|---|---:|---:|---:|---|
| `CBraMod_frozen` | `current_SASCert_SoftAR` | `targets_90_92_seeds_20_21` | 0.0124 | 0.0005 | 0.0277 | `not_promoted_calibration_and_macro_f1_weak` |
| `CBraMod_frozen` | `repaired_artifact_gate_physio` | `targets_90_92_seeds_20_21` | 0.0111 | 0.0426 | 0.0227 | `not_promoted_calibration_failed` |
| `CBraMod_frozen` | `repaired_temperature_scaled` | `targets_90_92_seeds_20_21` | 0.0111 | 0.0426 | 0.0221 | `not_promoted_temperature_scaling_failed_calibration_gate` |
| `ST-EEGFormer-small_source_tuned` | `SoftWeight_noReject` | `targets_90_109_seeds_20_24` | 0.0065 | 0.0064 | 0.0003 | `not_promoted_subject_seed_reliability_failed` |

## ST Utility Alignment

- Decision: `park_st_weighting_variants`
- Strong alignment found: `False`
- Top candidate-only feature: `clean_artifact_risk_raw_mean`
- Top candidate-only Spearman: `0.1168`

## Gates

| Gate | Passed |
|---|---:|
| `diagnostic_auc_pass` | `True` |
| `scalar_failure_pass` | `True` |
| `protocol_pass` | `True` |
| `weighting_policy_non_promotion_pass` | `True` |

## Decision

`write_diagnostic_certificate_paper_path`

Supported claim:

> SAS-Cert is currently supported as a diagnostic reliability certificate for EEG augmentation candidates.

Unsupported claim:

> Current SAS-Cert weighting/rejection policies are not supported as reliable deployable training methods.

Why:

Both backbones show meaningful diagnostic score structure, but CBraMod fails calibration after repair and ST fails subject/seed reliability plus utility alignment.

## Next Action

`prepare_diagnostic_certificate_paper_outline`
