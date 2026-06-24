# SAS-Cert-SoftAR-LS v1.1 Blueprint

## Unified Algorithm Logic

SAS-Cert-SoftAR-LS v1.1 should remain a risk-controlled
augmentation-utilization policy:

1. Safety Gate
2. Label-Preservation Evidence
3. Physiology/Style Plausibility
4. Utility Weight
5. Calibration-aware Training

The algorithm is not a flat sum of external modules. Each reference
tool supports one decision layer, while SAS-Cert keeps the decision
order and training policy.

## Layer 1: Safety Gate

Purpose: reject or downweight augmented trials that are likely
unsafe because of artifact or severe signal-quality failure.

Minimal v1.1:

- Primary implementation: existing rule artifact score using
  high-frequency energy, low-frequency drift, line-noise power,
  channel energy outliers, kurtosis, and skewness.
- Optional offline audit: Autoreject reject logs on MNE Epochs.
- Deferred offline audit: MNE-ICALabel component probabilities after
  fitted ICA, only after a version-matched MNE/ICALabel environment is prepared.
- Optional feature support: MNE-Features for trial-level statistics.

Policy:

- Highest artifact-risk decile is rejected: `w = 0`.
- Artifact risk is a qualification gate, not a positive additive
  score.

## Layer 2: Label-Preservation Evidence

Purpose: test whether `x_aug` still preserves the label semantics of
the original trial.

Inputs:

- ST-EEGFormer-small embedding.
- CBraMod embedding.
- Current classifier prediction on `x` and `x_aug`.

Evidence:

```text
E_embed = cosine(f(x), f(x_aug))
E_proto = cosine(f(x_aug), prototype_y)
        - max_{c != y} cosine(f(x_aug), prototype_c)
E_pred = -KL(p(.|x) || p(.|x_aug))
E_content = ranknorm(E_embed) + ranknorm(E_proto)
          + 0.5 * ranknorm(E_pred)
```

SCOPE/EEGTune are design references for agreement and prediction
stability, but their pseudo-label systems are not imported into
v1.1.

## Layer 3: Physiology/Style Plausibility

Purpose: ensure that label-like samples do not violate EEG
physiology or target-subject style.

Direct tools:

- pyRiemann for covariance matrices and Riemannian distances.
- MNE-Features for bandpower/statistical features.

Minimal v1.1:

```text
D_band = bandpower_deviation(x_aug, x_or_source_reference)
D_cov = riemannian_covariance_distance(x_aug, reference)
E_physio = 1 - ranknorm(D_band + D_cov)

style_target = target support mean/std + bandpower + covariance summary
style_aug = same summary for x_aug
D_style = distance(style_aug, style_target)
E_style = 1 - ranknorm(D_style)
```

Style remains auxiliary because previous audits showed that style can
be unstable across backbones and bad types.

## Layer 4: Utility Weight

For samples that pass the safety gate:

```text
score = E_content + E_physio + 0.5 * E_style
w = 0.2 + 0.8 * ranknorm(score)
```

For rejected samples:

```text
w = 0
```

## Layer 5: Calibration-aware Training

Minimal loss:

```text
L = CE(real_support, y; label_smoothing=0.10)
  + mean_i w_i * CE(x_aug_i, y_i; label_smoothing=0.10)
```

Label smoothing is fixed at `0.10` for the v1.1 comparison, matching
the current LS010 branch.

## Tools Not Entering v1.1 As Required Dependencies

- EEG-DLite: useful for later outlier/redundancy filtering, but its
  native workflow is data-distillation/pretraining-oriented and
  should not block v1.1.
- MOABB: useful for benchmark protocol and loaders, but not part of
  the core algorithm.
- Channel Reflection: useful knowledge-driven augmentation candidate
  for a later augmentation-pool expansion; not required for the
  initial v1.1 reliability policy.
- Braindecode: useful augmentation/baseline utilities, but current direct import
  is blocked by a heavier dependency stack; it should not define SAS-Cert's
  decision logic.
- MNE-ICALabel: useful artifact-probability reference, but current direct import
  is blocked by MNE API/version mismatch; use Autoreject plus rule/MNE-Features
  safety evidence first.
- HAPPE/PREP/ADJUST/ArtifactGen: cite or later only; do not force
  MATLAB/EEGLAB or heavy generative systems into v1.1.

## Next Experiment After User Approval

Dataset/backbone:

- ST-EEGFormer-small + PhysioNetMI left-vs-right MI, runs R04/R08/R12.

Four groups:

- NaiveAug_LS010
- ArtifactReject_LS010
- SoftWeight_noReject_LS010
- SAS-Cert-SoftAR-LS-v1.1

Promotion gate:

- v1.1 must beat or match the relevant baselines on balanced
  accuracy/Macro-F1 without worsening ECE/NLL/Brier, and it must
  improve subject/seed reliability rather than only mean metrics.
