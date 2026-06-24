# SAS-Cert Diagnostic Certificate Paper Outline

## Working Title

**SAS-Cert-EEG: Diagnostic Reliability Certificates for EEG Augmentation in Few-Shot Cross-Subject Adaptation**

## Current Core Claim

SAS-Cert is not yet a reliable deployable augmentation weighting policy.

It is currently supported as a diagnostic certificate that:

1. detects harmful EEG augmentation modes,
2. exposes score-direction failures,
3. separates diagnostic reliability from training utility,
4. explains why naive or scalar augmentation selection can be unsafe.

## Main Evidence Locked So Far

### One Dataset, Two Backbones

- Dataset: `PhysioNetMI`
- Task: left/right motor imagery, runs `R04/R08/R12`
- Backbones:
  - `CBraMod_frozen`
  - `ST-EEGFormer-small_source_tuned`

### Diagnostic AUC Evidence

| Backbone | Current SAS AUC | Component-Gated v1 AUC | Artifact-Gate Physio AUC |
|---|---:|---:|---:|
| `CBraMod_frozen` | 0.1969 | 0.8395 | 0.9022 |
| `ST-EEGFormer-small_source_tuned` | 0.1662 | 0.8395 | 0.9022 |

Interpretation:

- The old scalar score is directionally wrong on the mixed-bad PhysioNetMI pool.
- Component-gated and artifact-gate-physio diagnostics recover clean-vs-bad separation.

### Training Policy Evidence

| Backbone | Policy | Main Result | Promotion Decision |
|---|---|---|---|
| `CBraMod_frozen` | repaired artifact-gate physio | Macro-F1 `+4.26pp`, ECE `+2.27pp` | not promoted |
| `ST-EEGFormer-small_source_tuned` | SoftWeight no-reject | Macro-F1 `+0.64pp`, subject reliability failed | not promoted |
| `ST-EEGFormer-small_source_tuned` | utility alignment audit | max candidate-only Spearman `0.1168` | park weighting variants |

Interpretation:

- Diagnostic score quality does not guarantee deployable training utility.
- The contribution should avoid overclaiming accuracy improvement.

## Proposed Paper Structure

### 1. Introduction

Problem framing:

- EEG augmentation is not automatically reliable in few-shot cross-subject adaptation.
- Harmful augmentations can preserve superficial style while damaging task content, physiology, or artifact structure.
- Foundation model adaptation needs augmentation diagnostics, not only more augmented samples.

Core question:

```text
How can we decide whether an augmented EEG sample is a beneficial subject-style
variation or a harmful content/physiology/artifact distortion?
```

### 2. Related Work

Keep focused:

- EEG foundation models and few-shot adaptation.
- EEG augmentation and negative transfer.
- Style/content disentanglement in EEG.
- Artifact and physiology-aware reliability measures.

### 3. SAS-Cert Diagnostic Framework

Present four diagnostic axes:

- content stability,
- subject/style compatibility,
- physiological plausibility,
- artifact risk.

Important wording:

- Call the output a **certificate score/profile**, not a final training policy.
- Define scalar scores as diagnostic summaries, with direction audit required.

### 4. Experimental Protocol

Locked protocol:

- Dataset: PhysioNetMI.
- Backbones: CBraMod and ST-EEGFormer-small.
- Existing few-shot target-support setup.
- No target held-out leakage for score/rank/threshold/training selection.
- Bad augmentation pool used only for diagnostic separability.

### 5. Results

Result blocks:

1. scalar SAS score failure on PhysioNetMI,
2. component-gated and artifact-gate-physio diagnostic recovery,
3. CBraMod training-policy non-promotion due to calibration,
4. ST training-policy non-promotion due to subject/seed reliability,
5. utility-alignment audit showing no legal support/candidate predictor.

### 6. Discussion

Main message:

- Diagnostic certification and deployable augmentation selection are different problems.
- SAS-Cert currently solves the former more convincingly than the latter.
- This is still valuable because it prevents false confidence in naive augmentation policies.

Limitations:

- One main dataset in the locked diagnostic pack.
- Synthetic bad augmentation definitions still shape the diagnostic task.
- Training policies need a new calibration-aware or utility-aware hypothesis.

### 7. Next Work

Only after this paper path is coherent:

- Add calibration-aware training loss for CBraMod repaired score.
- Add support-only utility modeling if a new non-leaky predictor is found.
- Extend diagnostic certificate to a second dataset.

## Minimal Evidence Checklist

Checklist status:

- [x] Add one table of bad-type AUC by component from existing outputs.
- [x] Add one protocol-leakage audit table.
- [x] Add one figure-ready causal chain diagram:
  `diagnostic separability -> score direction -> training policy -> reliability`.
- [x] Add short definitions of all augmentation failure modes.
- [x] Verify every reported number can be traced to a file path.

Evidence file:

```text
docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md
```

Generated tables:

```text
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/bad_type_component_auc.csv
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/number_traceability.csv
```

## Current Decision

```text
write_diagnostic_certificate_paper_path
```

Do not restart weighting/gating experiments until the diagnostic certificate
paper outline and evidence checklist are complete.
