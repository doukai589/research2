# SAS-Cert v3 Training Plan

## Backbone Checkpoint

- Backbone: `ST-EEGFormer-small`
- Checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`

## Frozen Modules

- ST-EEGFormer-small backbone is frozen.

## Trainable Modules

- Classifier head only for OracleRiskReject.
- No CertAdapter.
- No calibrator unless OracleRiskReject passes recoverability criteria.

## Data Streams

| Stream | Oracle Route | Calibrator Training | Training | Final Test |
|---|---:|---:|---:|---:|
| source train | no | only if oracle succeeds | source head/style bank | no |
| validation subjects | no | threshold/calibrator only if oracle succeeds | no | no |
| target support | yes | no target-test leakage | yes | no |
| target test | no | no | no | yes |

## Oracle Risk Label

Risk label is generated from augmentation type assigned during candidate
generation:

- mild = 0
- risky = 1 for `strong_frequency_mask`, `strong_channel_dropout`,
  `emg_like_burst`, `eog_like_drift`, and `covariance_perturbation`

## Candidate Pool

- Fixed risk-mixed pool.
- 70% mild augmentation.
- 30% risky augmentation.

## Loss Function

- `RiskMixed_NaiveAug_LS010`:
  - `CE(real) + normalized CE(all augmented candidates)`
- `RiskMixed_OracleRiskReject_LS010`:
  - `CE(real) + normalized CE(mild augmented candidates)`
  - risky candidates are quarantined from supervised CE

## Baseline

- Primary baseline: `RiskMixed_NaiveAug_LS010`

## Checkpoint / Threshold Selection

- Fixed source-tuned checkpoint.
- Fixed epoch count.
- No target-test best epoch, best seed, threshold, prototype, or ranknorm.

## Target Test Use

- Target test is used only for final evaluation.
