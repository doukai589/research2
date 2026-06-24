# Failure Review: ST-EEGFormer-small PhysioNetMI SAS-Cert Full Matrix

## 1. Data Facts

Run scope:

- Targets: subjects `90-109`
- Seeds: `20,21,22,23,24`
- Rows: `400` group evaluations
- Protocol: frozen pretrained ST-EEGFormer-small feature extractor + feature head adaptation
- Groups: `NaiveAug_LS010`, `ArtifactReject_LS010`, `SoftWeight_noReject_LS010`, `SASCert_SoftAR_LS010`

Aggregate metrics:

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.5039 | 0.4921 | 0.2649 | 0.8854 | 0.6172 |
| ArtifactReject_LS010 | 0.5067 | 0.4948 | 0.2676 | 0.8840 | 0.6168 |
| SoftWeight_noReject_LS010 | 0.5106 | 0.4982 | 0.2642 | 0.8876 | 0.6187 |
| SASCert_SoftAR_LS010 | 0.5161 | 0.5037 | 0.2619 | 0.8842 | 0.6170 |

SASCert vs Naive:

| Metric | Delta |
|---|---:|
| Balanced accuracy | +0.0121 |
| Macro-F1 | +0.0116 |
| ECE | -0.0030 |
| NLL | -0.0011 |
| Brier | -0.0002 |
| Subject win rate Macro-F1 | 0.25 |
| Seed win rate Macro-F1 | 0.40 |

Data quality notes:

- No NaN/Inf loss or runtime stop occurred.
- Artifact feature kurtosis produced precision-loss warnings on near-constant samples; values are sanitized with `nan_to_num`.
- All groups are close to chance-level BAcc, which is the main abnormal finding.

## 2. Expected vs Actual

| Expectation | Actual Result | Verdict | Key Data |
|---|---|---|---|
| SASCert improves Macro-F1 over Naive by at least +0.005 | Improved by +0.0116 | Supported on mean | `0.5037 - 0.4921` |
| SASCert does not reduce BAcc | Improved by +0.0121 | Supported on mean | `0.5161 - 0.5039` |
| ECE does not worsen by more than +0.01 | Improved by -0.0030 | Supported | `0.2619 - 0.2649` |
| NLL does not worsen by more than +0.01 | Improved by -0.0011 | Supported | `0.8842 - 0.8854` |
| Brier does not worsen by more than +0.01 | Improved by -0.0002 | Supported | `0.6170 - 0.6172` |
| Subject win rate Macro-F1 >= 0.65 | 0.25 | Rejected | only 5/20 subjects majority-win |
| Seed win rate Macro-F1 >= 0.65 | 0.40 | Rejected | only 2/5 seeds majority-win |
| ST backbone provides a strong few-shot feature space | All groups near chance | Rejected/uncertain | best BAcc only `0.5161` |

## 3. Causal Chain

```text
ST pretrained features are class-separable for PhysioNetMI
  -> source-trained feature head learns a useful MI boundary
  -> target support + augmentation can adapt that boundary
  -> SAS-Cert ranks/weights better augmented samples
  -> average and subject-wise reliable adaptation improves
```

Evidence:

- `ST pretrained features are class-separable`: rejected/uncertain. All groups remain around BAcc `0.50-0.52`.
- `source-trained feature head learns useful boundary`: weak evidence. Naive BAcc only `0.5039`.
- `SAS-Cert ranks/weights better samples`: partially supported. Mean deltas are positive and calibration slightly improves.
- `subject-wise reliable adaptation improves`: rejected. Subject win rate `0.25`.

First broken link:

```text
ST pretrained frozen features -> useful MI few-shot feature space ✗
```

## 4. Possible Explanations

### Explanation A: Frozen pretrained ST features are not enough; supervised source tuning is required

Logic:

- Earlier ST full fine-tune reached BAcc `0.7669`.
- Current frozen-pretrained-feature protocol reaches only BAcc `0.5161`.
- The large gap suggests the supervised source/full fine-tuning stage created the useful MI representation.

What it explains:

- Why all four groups are near chance.
- Why SAS-Cert has positive but small mean effects.

Credibility: high.

Needed data:

- Repeat this runner using the previously source/validation-trained ST checkpoint as the frozen feature extractor.

### Explanation B: Current SAS-Cert score is directionally useful but unstable across subjects

Logic:

- Mean deltas are positive.
- Component ablation beats ArtifactReject and SoftWeight on mean Macro-F1.
- But subject win rate is only `0.25`.

What it explains:

- Why average improves but Go criteria fail.

Credibility: medium.

Needed data:

- Subject-wise score diagnostics: artifact/content score distributions and per-subject delta correlation.

### Explanation C: Current augmentation pool is too synthetic/stress-heavy for ST features

Logic:

- Candidate pool includes bad-content, bad-physio, and artifact stress samples.
- If base features are weak, adding stressful candidates may dominate the few-shot signal.

What it explains:

- Near-chance performance and unstable wins.

Credibility: medium-low until source-tuned feature test is done.

Needed data:

- Compare the same runner with clean/mild-only candidates after checking source-tuned features.

## 5. Adjusted Plan

Original plan:

```text
Pretrained ST frozen features + few-shot support augmentation
```

Adjusted next experiment:

```text
Use source/validation-trained ST checkpoint as feature extractor
  -> rerun the same SAS-Cert four groups
  -> keep dataset, subjects, seeds, augmentation, and locked SAS-Cert parameters unchanged
```

Why this next:

- It tests the first broken causal link directly.
- It reuses existing outputs:
  - `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- It does not add a new model, dataset, or hyperparameter search.

Decision:

```text
revise_training
```

Do not expand to emotion datasets or EEGPT. First verify whether source-supervised ST features restore a useful adaptation substrate.

## 6. Student-Facing Summary

The data say two things at once. SAS-Cert has a positive average signal on ST-EEGFormer-small: Macro-F1 improves by `+1.16pp`, BAcc by `+1.21pp`, and calibration does not worsen. But the result is not reliable enough because subject win rate is only `0.25` and seed win rate is only `0.40`.

The main issue is likely not that SAS-Cert is useless. The bigger problem is that this workbench runner used frozen pretrained ST features, while the earlier strong ST result came from supervised full fine-tuning. In this frozen-feature protocol all groups stay near chance, so the first hidden premise failed: the feature space itself is not yet strong enough for target few-shot adaptation.

The next best step is one focused rerun: use the already trained ST checkpoint from the full fine-tune run as the frozen feature extractor, then run the same four SAS-Cert groups without changing SAS-Cert parameters.

## 7. Focused Rerun Result

Rerun:

- Frozen feature extractor: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `st_source_tuned_full`
- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Rows: `400`

Aggregate metrics:

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7088 | 0.7045 | 0.2079 | 0.6853 | 0.4159 |
| ArtifactReject_LS010 | 0.7107 | 0.7064 | 0.2112 | 0.6854 | 0.4165 |
| SoftWeight_noReject_LS010 | 0.7153 | 0.7109 | 0.2082 | 0.6832 | 0.4103 |
| SASCert_SoftAR_LS010 | 0.7149 | 0.7108 | 0.2056 | 0.6810 | 0.4097 |

SASCert vs Naive:

| Metric | Delta |
|---|---:|
| Balanced accuracy | +0.0061 |
| Macro-F1 | +0.0063 |
| ECE | -0.0023 |
| NLL | -0.0044 |
| Brier | -0.0062 |
| Subject win rate Macro-F1 | 0.25 |
| Seed win rate Macro-F1 | 0.00 |

Updated interpretation:

- The first broken link was correctly identified: source-tuned ST features repair the near-chance feature-substrate problem.
- The locked SoftAR method still does not pass reliability criteria on ST because subject and seed win rates remain low.
- `SoftWeight_noReject_LS010` slightly exceeds SoftAR on mean Macro-F1, while SoftAR has better ECE/NLL/Brier.
- The next change should be method-local: diagnose whether artifact rejection is over-pruning useful augmented samples in the source-tuned ST feature space.

Updated decision:

```text
revise_method_after_source_tuned_rerun
```

Do not expand to a new dataset or backbone yet.
