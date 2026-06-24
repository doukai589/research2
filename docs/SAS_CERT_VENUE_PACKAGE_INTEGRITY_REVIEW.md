# SAS-Cert Venue Package Integrity Review

## Status

Passed with one environment limitation: the package is structurally ready for a
venue-template conversion pass, but local PDF compilation was not run because
`latexmk`, `pdflatex`, and `chktex` are not available in this environment.

## Reviewed Package

- Paper directory: `paper/sas_cert_diagnostic_certificate`
- Main source: `paper/sas_cert_diagnostic_certificate/main.tex`
- Bibliography: `paper/sas_cert_diagnostic_certificate/references.bib`
- Figures: `paper/sas_cert_diagnostic_certificate/figures`
- Tables: `paper/sas_cert_diagnostic_certificate/tables`
- Builder: `scripts/build_sas_cert_paper_package.py`

## Checks

| check | result | evidence |
| --- | --- | --- |
| Citation consistency | passed | 10 cited keys, 0 missing BibTeX entries |
| Bibliography filtering | passed | 10 package entries, 0 unused package entries |
| Figure paths | passed | 5 `\includesvg` paths, 0 missing SVG files |
| Table inputs | passed | 4 `\input{tables/...}` paths, 0 missing inputs |
| Document boundary | passed | exactly one `\begin{document}` and one `\end{document}` |
| List environments | passed | 4 itemize begins and 4 itemize ends |
| Verbatim environments | passed | 2 verbatim begins and 2 verbatim ends |
| Markdown residue | passed | no `##`, markdown table separators, or code fences in `main.tex` |
| Internal path exposure | passed | no `workbench/` or `outputs/runs/` paths in `main.tex` |
| Local PDF compilation | not run | `latexmk` and `pdflatex` unavailable |
| Local LaTeX lint | not run | `chktex` unavailable |

## Fixes Applied During Review

- Updated `scripts/build_sas_cert_paper_package.py` to remove source-authored
  section numbers before LaTeX auto-numbering.
- Updated generated inline tables to use text-safe column alignment and
  `\resizebox{\linewidth}{!}{...}` to reduce immediate table overflow risk.
- Added `graphicx` to `main.tex` through the package generator because
  `\resizebox` requires it.
- Regenerated the full paper package after these fixes.

## Remaining Venue-Template Gaps

- The current package uses a generic `article` class. A venue-specific class
  such as IEEE, ACM, Springer, or arXiv style has not yet been selected.
- SVG figures require the LaTeX `svg` package and normally need
  `-shell-escape`. Some venues prefer PDF/PNG/EPS figure assets; conversion
  should be handled in the venue-template pass.
- Tables are currently compact and structurally valid, but final layout must be
  checked inside the target venue template.
- The author block is still a placeholder: `SAS-Cert EEG Project`.

## Claim Boundary Audit

No new experiments, training runs, data copies, or scientific claims were
introduced. The package still preserves the current claim boundary:

- Supported: SAS-Cert as a diagnostic reliability certificate for EEG
  augmentation candidates.
- Supported: component-level auditing reveals scalar score-direction failures.
- Not promoted: current SAS-Cert weighting/rejection policies as deployable
  training methods.

## Decision

`proceed_to_venue_template_selection`

The paper package is ready for a venue/template selection step. Do not add new
experiments before that step unless a specific venue review exposes a claim or
evidence gap that cannot be fixed by formatting alone.
