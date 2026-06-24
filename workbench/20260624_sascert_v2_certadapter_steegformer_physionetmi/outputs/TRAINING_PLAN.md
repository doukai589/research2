# SAS-Cert v2 Training Plan

## Backbone Checkpoint

- Backbone: `ST-EEGFormer-small`
- Checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`

## Frozen Modules

- ST-EEGFormer-small backbone is frozen.
- Frozen ST embeddings are cached and reused.

## Trainable Modules

- Source-initialized classifier head.
- Task-Prior Head trained from source train plus target support clean samples.
- Multi-prototype bank is constructed from source train plus target support
  clean embeddings and is not fit from target test.
- CertAdapter and classifier head for `SAS-Cert-v2-full`.

## Data Streams

| Stream | Prototype | Certificate | Training | Validation | Final Test |
|---|---:|---:|---:|---:|---:|
| source train | yes | no | task-prior/source head | no | no |
| validation subjects | no | no | no | not used for selection | no |
| target support | yes | yes | yes | no | no |
| target test | no | no | no | no | yes |

## Augmentation Candidate Pool

- Regular pool uses existing mild augmentations.
- Risk-mixed pool uses a fixed 70% mild / 30% risky candidate mix.
- Risky augmentations include strong frequency mask, strong channel dropout,
  EMG-like burst, EOG-like drift, and covariance perturbation.

## Certificate Calculation

For every target-support sample `x` and augmented candidate `x_aug`:

- `E_embed = cosine(f(x), f(x_aug))`
- `E_prior` from Task-Prior Head original-class margin on `x_aug`
- `E_proto` from max own-class multi-prototype similarity minus max other-class
  multi-prototype similarity
- prior agreement: Task-Prior predicted label equals original label
- prototype agreement: prototype predicted label equals original label
- `gamma` is a bounded agreement-modulated confidence from `E_embed`,
  `E_prior`, and `E_proto`

Artifact, physio, and style scores are diagnostic-only.

## Loss Function

- `L_real = CE(real, y, label_smoothing=0.10)`
- `L_aug = sum(gamma_i * CE(x_aug_i, y_i)) / (sum(gamma_i) + eps)`
- `L_cons = gamma_i * KL(p(x), p(x_aug))`
- `L_proto = gamma_i * prototype_margin_loss(h_cert(x_aug), y_i)`
- `L = L_real + 1.0 * L_aug + 0.2 * L_cons + 0.2 * L_proto`

## Checkpoint Selection

- Fixed source-tuned ST checkpoint.
- Fixed epoch count.
- No target-test best epoch or best seed selection.

## Target Test Use

- Target test is used only once per fold for final metric evaluation.
