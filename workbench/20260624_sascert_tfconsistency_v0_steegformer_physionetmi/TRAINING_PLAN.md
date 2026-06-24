# SAS-Cert-TFConsistency v0 Training Plan

## Backbone Checkpoint

- Backbone: `ST-EEGFormer-small`
- Checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`

## Frozen Modules

- ST-EEGFormer-small backbone is frozen.

## Trainable Modules

- Classifier head only.
- No CertAdapter.
- No backbone fine-tuning.

## Data Streams

| Stream | Prototype | Certificate | Augmentation | Training | Validation | Final Test |
|---|---:|---:|---:|---:|---:|---:|
| source train | yes | no | style/reference bank | source head init | no | no |
| validation subjects | no | no | no | no | not used for selection | no |
| target support | yes | yes | yes | yes | no | no |
| target test | no | no | no | no | no | yes |

## Target Test Use

- Target test is used only for final metrics.
- It is not used for prototype, ranknorm, threshold, route, best epoch, or best seed.

## Augmentation Views

Regular TF views:

- `weak_frequency_mask`
- `weak_time_shift`
- `weak_amplitude_scaling`
- `same_class_frequency_mixup`
- `frequency_mixup`

Risk-mixed TF views additionally include:

- `strong_frequency_mask`
- `wrong_class_frequency_mixup`
- `emg_like_burst`
- `eog_like_drift`

CSDA / DWT / HHT:

- `pywt` is unavailable, so DWTaug smoke failed and is not used in the main run.
- HHTAug is audit-only in this round.

## SAS-Cert Certificate

- `E_embed = cosine(f(x), f(v))`
- `E_proto = cosine(f(v), prototype_y) - max_other cosine(f(v), prototype_other)`
- `content_q = ranknorm(E_embed + E_proto)` within the training candidate pool
- `risk_q = ranknorm(artifact_score + physio_deviation)` within the training candidate pool
- style distance is diagnostic-only in v0

## Route Rule

Fixed v0 thresholds:

- supervised:
  - `content_q >= 0.67`
  - `risk_q <= 0.50`
- consistency:
  - `content_q >= 0.50`
  - `risk_q <= 0.85`
  - not supervised
- quarantine:
  - `content_q < 0.50`
  - or `risk_q > 0.85`
  - or NaN/Inf/extreme artifact

## Loss Function

- `L_real = CE(clean, y, label_smoothing=0.10)`
- `L_supervised = CE(supervised_view, y, label_smoothing=0.10)`
- `L_consistency = KL(stopgrad(p_clean) || p_consistency_view)`
- `L = L_real + 1.0 * L_supervised + 2.0 * L_consistency`
- Prototype loss is diagnostic-only in v0 because the backbone is frozen and
  only the classifier head is trainable.

## Baselines

Regular TF pool:

- `RealOnly_LS010`
- `NaiveTF-Aug_LS010`
- `AugMixTF_LS010`
- `SAS-Cert-TFConsistency_v0`

Risk-mixed TF pool:

- `RiskMixed_NaiveTF-Aug_LS010`
- `RiskMixed_AugMixTF_LS010`
- `RiskMixed_SAS-Cert-TFConsistency_v0`
