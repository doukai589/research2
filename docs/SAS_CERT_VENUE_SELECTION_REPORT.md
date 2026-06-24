# SAS-Cert Venue Selection Report

## Status

Completed default venue-route selection and generated an arXiv-first venue
package.

## Selected Route

`arxiv_article_first`

This route is selected as the default because no specific venue preference has
been provided yet. It is the lowest-risk next step for a long-running research
project: it preserves the current article structure, keeps the diagnostic claim
boundary intact, and avoids prematurely forcing the paper into IEEE/ACM/Springer
layout constraints before author metadata and final target venue are decided.

## Candidate Route Comparison

| route | current fit | risk | decision |
| --- | --- | --- | --- |
| arxiv_article_first | high | requires PDF/PNG figure conversion before clean upload | selected |
| ieee_conference_style | medium | two-column table/figure compression could distort readability | defer until user selects IEEE venue |
| acm_conference_style | medium | requires ACM metadata and rights blocks | defer until user selects ACM venue |
| journal_template_after_author_decision | medium | needs concrete journal and author metadata | defer |

## Generated Venue Package

- Venue package directory: `paper/sas_cert_diagnostic_certificate_venue`
- Main source: `paper/sas_cert_diagnostic_certificate_venue/main.tex`
- Bibliography: `paper/sas_cert_diagnostic_certificate_venue/references.bib`
- Table inputs: `paper/sas_cert_diagnostic_certificate_venue/tables`
- SVG originals: `paper/sas_cert_diagnostic_certificate_venue/figures_svg_original`
- Figure target directory: `paper/sas_cert_diagnostic_certificate_venue/figures`
- Builder: `scripts/build_sas_cert_arxiv_package.py`

## Figure Format Decision

The arXiv-first `main.tex` no longer depends on the LaTeX `svg` package. It
expects PDF figures in `figures/` and uses guarded placeholders when those PDFs
are missing.

No SVG-to-PDF conversion was performed because the local environment lacks all
checked conversion tools:

- `inkscape`: unavailable
- `rsvg-convert`: unavailable
- Python `cairosvg`: unavailable

The original SVG figures were preserved in `figures_svg_original/` for a later
conversion pass.

## Integrity Checks

| check | result |
| --- | --- |
| Missing BibTeX entries | none |
| Unused package BibTeX entries | none |
| `svg` LaTeX package dependency | removed |
| `\includegraphics` figure slots | 5 |
| guarded missing-figure fallbacks | 5 |
| Internal `workbench/` paths in venue `main.tex` | none |
| Internal `outputs/runs/` paths in venue `main.tex` | none |
| `[RE...]` placeholders in venue `main.tex` | none |

## Author Metadata Gaps

- Author list is still `SAS-Cert EEG Project`.
- Affiliations are not filled.
- Corresponding author metadata is not filled.
- Acknowledgements and funding statements are not filled.
- Competing-interest / ethics statements depend on final venue policy.

## Claim Boundary

No experiments, training runs, data copies, or scientific claims were added.
The selected venue route preserves the existing claim boundary:

- SAS-Cert is supported as a diagnostic reliability certificate for EEG
  augmentation candidates.
- Component-level auditing exposes scalar score-direction failures.
- Current weighting/rejection policies are not promoted as deployable training
  methods.

## Decision

`proceed_to_figure_conversion_and_compile_env`

Before a clean arXiv upload or venue submission, the project needs either:

1. SVG-to-PDF conversion tooling installed, or
2. regenerated figures directly in PDF/PNG format from the existing evidence
   tables.

No new experiments are needed for this step.
