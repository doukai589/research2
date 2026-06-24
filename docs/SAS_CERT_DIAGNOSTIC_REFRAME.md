# SAS-Cert Diagnostic Reframe

## Why The Route Changes

The current two-backbone, one-dataset evidence says SAS-Cert is useful, but not
in the original deployable-weighting form.

The original hope was:

```text
detect bad augmentation
  -> rank useful candidates
  -> weight/reject candidates
  -> improve few-shot target adaptation reliably
```

The evidence now supports the first part much more strongly than the last part.

## Evidence

### CBraMod + PhysioNetMI

The repaired CBraMod score showed that certificate direction matters:

| Result | Value |
|---|---:|
| current scalar SAS clean-vs-bad AUC | 0.1969 |
| direction-fixed total AUC | 0.8911 |
| artifact-gate physio AUC | 0.9022 |
| repaired Macro-F1 delta vs Naive | +4.26pp |
| repaired ECE delta vs Naive | +2.27pp |

Interpretation:

- The diagnostic score can expose a broken scalar direction.
- Repaired weighting can improve classification.
- Calibration failure prevents promotion to a reliable training method.

### ST-EEGFormer-small + PhysioNetMI

Source-tuned ST features created a usable adaptation substrate, but weighting
variants were unreliable.

| Branch | Key Result | Decision |
|---|---|---|
| `SASCert_SoftAR_LS010` | Macro-F1 `+0.63pp` vs Naive, but subject/seed reliability failed | not promoted |
| `SoftWeight_noReject_LS010` | Macro-F1 `+0.64pp` vs Naive, ECE stable, but majority-seed subject win rate `0.15` | not promoted |
| `ComponentGatedV1_LS010` | score-only AUC `0.8395`, but mini training Macro-F1 `-0.46pp` vs Naive | not expanded |
| utility-alignment audit | max candidate-only Spearman `0.1168` | park ST weighting variants |

Interpretation:

- Candidate diagnostics can be meaningful without becoming reliable training
  weights.
- No-reject weighting has a real average signal, but it is too heterogeneous
  for a locked method claim.

## Reframed Scientific Claim

SAS-Cert should currently be framed as:

> A diagnostic reliability certificate for EEG augmentation candidates that
> detects harmful augmentation modes, exposes score-direction failures, and
> explains why naive augmentation policies can be unsafe in few-shot
> cross-subject EEG adaptation.

It should not yet be framed as:

> A deployable augmentation-selection policy that reliably improves adaptation
> across target subjects and seeds.

## What Remains Valid

- The scientific problem remains strong: EEG augmentation can hurt because it
  may change task content, physiology, style, or artifact structure.
- The multi-component certificate framing remains useful.
- Artifact/physiology diagnostics are especially informative.
- Score direction should be audited per backbone/dataset before any training
  policy is trusted.

## What Must Be Paused

- Hard Top50 selection.
- ST weighting variants.
- CBraMod full PhysioNetMI expansion without a new calibration hypothesis.
- New datasets before the diagnostic certificate package is stable.
- More gate/threshold search on the current outputs.

## Next MVE

The next MVE should be a diagnostic package, not a training expansion.

Name:

```text
sas_cert_diagnostic_certificate_pack_physionetmi
```

Purpose:

```text
Lock the diagnostic claim on PhysioNetMI with CBraMod and ST-EEGFormer-small:
SAS-Cert detects bad augmentation modes and score-direction failures, while
current weighting/rejection policies remain non-promoted.
```

Allowed:

- Existing score rows and metrics.
- Cross-backbone component AUC tables.
- Failure-mode taxonomy.
- Protocol/leakage audit.
- A concise diagnostic report suitable for paper planning.

Forbidden:

- New training.
- New backbone.
- New dataset.
- New thresholds selected from target test outcomes.

Go criteria:

| Requirement | Gate |
|---|---|
| Two-backbone diagnostic score evidence is reproducible | component-gated AUC or artifact-gate-physio AUC remains `>=0.80` |
| Score-direction failure is clearly documented | current scalar SAS AUC `<0.50` on PhysioNetMI mixed-bad pool |
| Training-policy non-promotion is explicit | ST and CBraMod weighting decisions are marked non-promoted |
| Protocol is clean | no raw data copy; no target held-out labels used for score/rank/threshold selection |

Decision after next MVE:

- If the diagnostic package is coherent, write the paper around diagnostic
  certification and reliability analysis.
- If it is not coherent, park SAS-Cert as an internal diagnostic tool and stop
  adding augmentation-selection experiments.
