# SAS-Cert Submission Package Quality Review

## Review Scope

Reviewed package:

- `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md`
- `docs/SAS_CERT_BIBLIOGRAPHY_TRACE.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_asset_manifest.json`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/`

Review config:

```text
configs/experiments/submission_package_quality_review.yaml
```

## Executive Verdict

```text
submission_package_review_passed_with_formatting_and_bibtex_gaps
```

The submission-preparation package is complete enough to proceed to formatting
and bibliography cleanup. It should not trigger new experiments. The current
remaining gaps are manuscript-production issues: exact BibTeX resolution,
LaTeX equation conversion, formal ST-EEGFormer-small citation/source recording,
and figure/table styling.

## Verification Results

| Check | Verdict | Evidence |
|---|---|---|
| Manifest outputs exist | passed | `missing=[]`; manifest lists 6 tables and 5 figures. |
| SVG figures parse | passed | All 5 SVG files parse with `xml.etree.ElementTree`. |
| Submission draft removed project-management notes | passed | No `Citation placeholders`, `Evidence sources`, internal output paths, or `Evidence Index` remain in the submission draft. |
| No new experiments introduced | passed | Manifest constraints: `new_experiments=false`, `new_training=false`, `new_claims=false`. |
| Weighting/rejection policy not promoted | passed | Draft states diagnostic gains do not justify deployable training methods; tables mark unsupported claims as `do not claim`. |
| Figures/tables match plan | passed | Figure 1-5 and Table 1-5 plus appendix audit were generated. |

## Generated Assets

### Draft and Trace

- Submission draft:
  `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md`
- Bibliography trace:
  `docs/SAS_CERT_BIBLIOGRAPHY_TRACE.md`
- Asset manifest:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_asset_manifest.json`

### Tables

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table1_protocol_and_claim_boundary.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table2_diagnostic_auc_summary.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table3_bad_type_component_auc.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table4_training_policy_non_promotion.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table5_claim_support.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/appendix_protocol_leakage_audit.md`

### Figures

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure1_certificate_overview.svg`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure2_diagnostic_auc.svg`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure3_component_specificity_heatmap.svg`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure4_training_policy_non_promotion.svg`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure5_causal_chain.svg`

## Claim Boundary Audit

Allowed main claim:

```text
SAS-Cert is currently supported as a diagnostic reliability certificate for
EEG augmentation candidates.
```

Still prohibited:

```text
SAS-Cert reliably improves few-shot accuracy across subjects and seeds.
SAS-Cert is ready as a deployable augmentation-selection method.
```

The prohibited claims only appear in claim-boundary tables with `do not claim`
labels. They are not asserted in the submission draft.

## Remaining Production Gaps

| Gap | Priority | Next action |
|---|---:|---|
| BibTeX placeholders unresolved | high | Create `docs/SAS_CERT_REFERENCES.bib` or a citation trace table with exact titles/authors/years. |
| ST-EEGFormer-small formal source missing | high | Record repository/paper source before final bibliography. |
| Equations are plaintext | medium | Convert certificate definitions to LaTeX display equations in a submission-polished draft. |
| Figure styling is draft-quality | medium | Improve SVG fonts, sizes, axis labels, and panel order without changing data. |
| Table 3 is too long for main text | medium | Keep compact summary in main text and move full component AUC table to appendix. |
| Venue formatting not selected | low | Decide after bibliography and figures are clean. |

## Decision

```text
proceed_to_bibtex_latex_figure_polish
```

Do not run new experiments yet. The next work should stay in manuscript
production:

1. Resolve bibliography placeholders and formal source records.
2. Convert plaintext equations into LaTeX notation.
3. Polish generated figures/tables for readability while preserving existing
   evidence and claim boundaries.
