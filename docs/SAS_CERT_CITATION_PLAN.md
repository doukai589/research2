# SAS-Cert Citation Plan

## Purpose

This file maps the current SAS-Cert diagnostic-certificate manuscript to local reference anchors. It is a writing plan, not a new evidence source. Numerical claims must still trace to project outputs listed in the claim-support table.

Primary manuscript draft:

```text
docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md
```

Project evidence table:

```text
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv
```

## Citation Boundary

Use references to motivate the scientific problem and method design. Do not use references to replace local evidence for SAS-Cert performance.

Allowed:

- EEG foundation models motivate frozen-backbone evaluation.
- Style/content and subject-latent papers motivate certificate axes.
- MI physiological prior papers motivate mu/beta and covariance checks.
- Artifact papers motivate artifact-risk as an independent component.
- Low-label/foundation model papers motivate reliability metrics beyond accuracy.

Not allowed:

- Do not claim SAS-Cert deployable training success from external papers.
- Do not cite unrelated high-accuracy papers as if they validate SAS-Cert.
- Do not cite style/content augmentation work as proof that current SAS-Cert weighting works.

## Core Citation Map

| Manuscript location | Citation placeholder | Local source | Use |
|---|---|---|---|
| Abstract / Introduction | `[Wang2025CBraMod]` | `docs/references/Wang 等 - 2025 - CBraMod A Criss-Cross Brain Foundation Model for EEG Decoding.pdf` | CBraMod as one frozen EEG foundation backbone. |
| Introduction / protocol context | `[Liu2026MIRepNet]` | `docs/references/Liu 等 - 2026 - MIRepNet A pipeline and pre-trained model for EEG-based motor imagery classification.pdf` | Motor-imagery reproduction context; parked baseline branch, not main evidence. |
| Introduction | `[RE199]` | `选题.txt` lines around foundation-model marginal gains | Motivate why reliability/certification is more appropriate than accuracy-chasing. |
| Introduction / related work | `[RE132]` | `选题.txt` low-label LaBraM+LoRA note | Motivate low-label adaptation and parameter-efficient context; not used as current protocol. |
| Related work / method | `[RE309]` | `选题.txt`, `参考论文集/论文集_004.txt` JSCCRA entry | Direct style/content augmentation precursor. |
| Related work / method | `[RE334]` | `选题.txt`, `参考论文集/论文集_004.txt` | Evidence that subject and content latents can be separable. |
| Content / physiology method | `[RE167]` | `选题.txt` MI frequency-time prior note | Motivate mu/beta and MI physiological constraints. |
| Style method | `[RE181]` | `选题.txt` RCC covariance note | Motivate covariance/correlation style features. |
| Artifact method | `[RE333]` | `参考论文集/论文集_004.txt` EEGANet entry | Motivate artifact contamination as a major EEG reliability issue. |
| Artifact method | `[RE342]` | `参考论文集/论文集_004.txt` automatic ICA artifact classifier entry | Motivate subject-independent artifact-risk scoring and artifact-specific components. |
| Prediction-stability related work | `[RE185]` | `选题.txt` EEGTune note | Contrast prediction stability with multi-component certificate profiles. |
| Reliability discussion | `[RE332]` | `选题.txt` data augmentation small-gain note | Motivate small but reliability-relevant gains and caution around accuracy-only claims. |

## Section-by-Section Citation Insertions

### Abstract

Suggested placeholders:

```text
... using CBraMod and ST-EEGFormer-small as two frozen EEG foundation backbones [Wang2025CBraMod, RE199] ...
```

Keep the abstract evidence numeric claims tied to local outputs, not to citations.

### Introduction

Insert:

- EEG foundation models have reusable representations but do not remove adaptation reliability risks: `[Wang2025CBraMod, RE199, RE132]`.
- Augmentation can conflate subject style and task content: `[RE309, RE334]`.
- EEG artifact and non-stationarity motivate reliability screening: `[RE333, RE342]`.

### Problem Definition

Insert:

- Style/content split: `[RE309, RE334]`.
- MI content and physiology: `[RE167]`.
- Covariance style: `[RE181]`.
- Artifact risk: `[RE333, RE342]`.

### Diagnostic Certificate Formulation

Insert:

- Frozen EEG encoder context: `[Wang2025CBraMod]`.
- Prediction consistency as an incomplete predecessor to multi-axis stability: `[RE185]`.
- Artifact-risk scoring as independent dimension: `[RE342]`.

### Experimental Protocol

Insert:

- CBraMod source: `[Wang2025CBraMod]`.
- MI protocol context and reproduction caution: `[Liu2026MIRepNet]`.

Do not cite MIRepNet as a main baseline result unless the text explicitly says it is parked/archival context.

### Discussion

Insert:

- Limited marginal gains and reliability framing: `[RE199, RE132, RE332]`.
- Need for component-level reliability rather than single scalar: `[RE309, RE333, RE342]`.

## Evidence Versus Literature Claims

| Claim type | Source of truth | Allowed citation role |
|---|---|---|
| SAS-Cert scalar AUC failure | Project diagnostic AUC CSV | None required; citations only motivate why auditing matters. |
| Component-gated AUC success | Project diagnostic AUC CSV | Literature motivates components, not result. |
| Training policy non-promotion | Project training policy table | Literature can explain why reliability metrics matter. |
| Style/content separation | Literature + method rationale | Do not claim local latent disentanglement unless measured. |
| Artifact-risk importance | Literature + local bad-artifact AUC | Use both: literature for motivation, local AUC for evidence. |

## Citation Cleanup To Do Later

- Replace placeholders with full BibTeX keys after bibliography extraction.
- Verify exact titles/authors/years for `RE199`, `RE132`, `RE309`, `RE334`, `RE167`, `RE181`, `RE333`, `RE342`, `RE185`, and `RE332`.
- Add BibTeX entries for CBraMod and MIRepNet PDFs from `docs/references/`.
- Keep `ST-EEGFormer-small` cited only after its local repository/paper source is formally recorded.
