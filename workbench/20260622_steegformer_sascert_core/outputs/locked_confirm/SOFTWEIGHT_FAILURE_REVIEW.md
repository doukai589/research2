# SoftWeight No-Reject Failure Review

## 1. Facts Only

Scope:

- Dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`
- Backbone: frozen source-tuned `ST-EEGFormer-small`
- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Pairs: `100`
- Support: `5-shot` per class

Mean metrics:

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7088 | 0.7045 | 0.2079 | 0.6853 | 0.4159 |
| `SoftWeight_noReject_LS010` | 0.7153 | 0.7109 | 0.2082 | 0.6832 | 0.4103 |
| `SASCert_SoftAR_LS010` | 0.7149 | 0.7108 | 0.2056 | 0.6810 | 0.4097 |

SoftWeight vs Naive:

- Delta BAcc: `+0.0065`
- Delta Macro-F1: `+0.0064`
- Delta ECE: `+0.0003`
- Delta NLL: `-0.0022`
- Delta Brier: `-0.0057`
- Positive subject mean-delta rate: `0.60`
- Majority-seed subject win rate: `0.15`
- Seed win rate: `0.00`

Notable heterogeneity:

- Best target subjects:
  - Subject `106`: Macro-F1 `+0.0408`, seed-positive fraction `1.00`
  - Subject `108`: Macro-F1 `+0.0397`, seed-positive fraction `0.80`
  - Subject `98`: Macro-F1 `+0.0258`, seed-positive fraction `0.40`
- Worst target subjects:
  - Subject `105`: Macro-F1 `-0.0174`, seed-positive fraction `0.20`
  - Subject `93`: Macro-F1 `-0.0135`, seed-positive fraction `0.20`
  - Subject `102`: Macro-F1 `-0.0119`, seed-positive fraction `0.20`

No NaN/OOM/runtime deviation was found in the locked confirmation outputs.

## 2. Expected vs Actual

| Expectation | Actual Result | Verdict | Key Data |
|---|---|---|---|
| Soft weighting should improve mean Macro-F1 by at least `+0.5pp` vs Naive | Mean Macro-F1 improved by `+0.64pp` | supported | `+0.0064` |
| BAcc should not meaningfully drop | BAcc improved by `+0.65pp` | supported | `+0.0065` |
| Calibration should not degrade beyond `+0.01` ECE/NLL/Brier | ECE `+0.0003`, NLL `-0.0022`, Brier `-0.0057` | supported | all within gate |
| Subject win rate should be at least `0.65` | Majority-seed subject win rate was `0.15`; positive-mean subject rate was `0.60` | rejected | below `0.65` |
| Seed win rate should be at least `0.65` | Seed win rate was `0.00` | rejected | below `0.65` |
| SoftWeight should not underperform SoftAR on Macro-F1 | SoftWeight was nearly tied and slightly higher | supported but weak | `+0.0001` Macro-F1 |

## 3. Causal Chain

```text
Bad/clean augmentation is separable
  → cert score ranks useful samples higher
  → soft weighting changes the training signal
  → few-shot target adaptation improves
  → calibration does not degrade
```

Link status:

| Link | Status | Evidence |
|---|---|---|
| Bad/clean augmentation is separable | supported for diagnostics | component-gated AUC `0.8395`, artifact-gate-physio AUC `0.9022` |
| Cert score ranks useful samples higher | uncertain | clean-vs-bad AUC is strong, but utility varies across target subjects |
| Soft weighting changes training signal | supported | mean Macro-F1 `+0.64pp`, BAcc `+0.65pp` |
| Few-shot target adaptation improves | partially supported | mean improves, but majority-seed subject win rate only `0.15` |
| Calibration does not degrade | supported | ECE `+0.0003`, NLL/Brier improved |

First broken link:

```text
cert score ranks useful samples higher
  → soft weighting changes training signal reliably across subjects
```

The certificate is not simply wrong. The hidden premise that clean/bad
diagnostic quality translates into stable training utility is not yet supported.

## 4. Possible Explanations

### Explanation A: Clean-vs-bad separability is not the same as subject-specific utility

Logic:

- The certificate can identify synthetic bad samples, but the remaining useful
  samples may help only some target subjects.
- This explains why score-only AUC is high while training wins are concentrated
  in subjects `106` and `108`.

What it explains:

- Mean gain exists.
- Subject reliability is weak.
- Component-gated detector can be strong but still fail training.

What it does not explain:

- Why SoftWeight no-reject slightly beats SoftAR even though SoftAR has better
  ECE/NLL/Brier.

Distinguishing measurement:

- Per-fold correlation between candidate score summaries, support difficulty,
  and SoftWeight delta Macro-F1.

Credibility: high.

### Explanation B: Artifact rejection removes candidates that are harmful in diagnostics but useful as regularization

Logic:

- ST-EEGFormer may benefit from a broad augmentation cloud even when some
  candidates look artifact-risky.
- This explains why no-reject soft weighting is the best mean classification
  group while artifact-gated methods are weaker.

What it explains:

- SoftWeight no-reject beats ComponentGatedV1 and ArtifactGatePhysio in the mini
  training check.
- SoftAR has better calibration but not better Macro-F1.

What it does not explain:

- Seed win rate remains `0.00`.

Distinguishing measurement:

- Compare fold-level gain with artifact-risk distribution and rejection count.

Credibility: medium-high.

### Explanation C: The head-only few-shot training objective is too sensitive to duplicated support-derived candidates

Logic:

- Every augmented candidate is derived from very small target support.
- Soft weighting may help on average but amplify fold-specific noise.

What it explains:

- Strong seed instability.
- Positive mean effect but low majority-seed subject win rate.

What it does not explain:

- Why subjects `106` and `108` benefit strongly and consistently.

Distinguishing measurement:

- Candidate diversity/duplicate concentration versus fold delta Macro-F1.

Credibility: medium.

## 5. Adjustment

Original plan:

```text
Promote SoftWeight no-reject if it beats Naive on mean metrics and reliability
gates.
```

Adjusted plan:

```text
Do not promote SoftWeight no-reject. Treat it as the simplest positive ST
training candidate, then run one existing-output utility-alignment audit to
identify whether the instability is due to score/utility mismatch or training
objective sensitivity.
```

Next focused experiment:

```text
support_candidate_utility_alignment_audit
```

Inputs:

- `st_source_tuned_full` metrics
- `st_source_tuned_full` score rows
- support/candidate summary statistics only

Allowed:

- Existing-output analysis
- Per-fold correlations and stratification
- No new model training

Forbidden:

- No new backbone
- No new dataset
- No target held-out labels for threshold/rule selection
- No additional gate search before this audit is interpreted

Updated Go/No-Go for the next audit:

| Result | Decision |
|---|---|
| Support/candidate features explain SoftWeight gains clearly | design one locked support-only routing or utility-aware weighting rule |
| No clear alignment | park ST weighting variants and frame SAS-Cert as diagnostic certification |
| Alignment only for weak target subjects | revise hypothesis toward subject-difficulty-conditioned augmentation utility |

## 6. Summary

The data say that `SoftWeight_noReject_LS010` has a real average benefit, but
not a reliable enough one. The method improves Macro-F1 by about `+0.64pp` and
BAcc by about `+0.65pp`, while keeping calibration essentially stable. That is
not noise-free failure; it is a heterogeneous effect.

The original hypothesis is partly right: weighting augmentation candidates can
help ST-EEGFormer few-shot adaptation. The part that failed is the implicit
assumption that a certificate score that separates clean/bad candidates will
also produce stable training utility across subjects and seeds.

The next best step is one focused utility-alignment audit, not another training
expansion. If that audit cannot explain who benefits and why, this branch should
be parked as a diagnostic observation rather than promoted as a deployable
method.
