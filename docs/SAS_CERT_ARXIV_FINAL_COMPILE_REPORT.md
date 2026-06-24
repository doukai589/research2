# SAS-Cert arXiv Final Compile Report

## Status

`compiled_successfully_with_nonblocking_chktex_style_warnings`

The arXiv-first SAS-Cert venue package was compiled locally with `latexmk` and
`pdflatex`. The generated PDF exists and includes all five PDF figures.

## Compile Environment

- OS: Ubuntu 20.04.4 LTS
- User: `root`
- Tooling route: local LaTeX toolchain installed with `apt-get`
- Installed/verified core tools:
  - `latexmk` 4.67
  - `pdflatex` pdfTeX 3.14159265-2.6-1.40.20
  - `bibtex` 0.99d
  - `chktex` 1.7.6

Installed package group:

```text
latexmk texlive-latex-base texlive-latex-recommended
texlive-latex-extra texlive-fonts-recommended chktex
```

## Compile Command

Run from:

```text
paper/sas_cert_diagnostic_certificate_venue
```

Command:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Output

- PDF: `paper/sas_cert_diagnostic_certificate_venue/main.pdf`
- PDF size: `271202` bytes
- PDF header: `%PDF-1.5`
- PDF pages: `11`
- Bibliography file generated: `paper/sas_cert_diagnostic_certificate_venue/main.bbl`
- Log file: `paper/sas_cert_diagnostic_certificate_venue/main.log`

## Log Scan

| item | count |
| --- | ---: |
| `LaTeX Warning` | 0 |
| `Overfull` | 0 |
| `Undefined` | 0 |
| `Error` | 0 |
| `Emergency stop` | 0 |
| `Fatal` | 0 |

The final log includes:

```text
Output written on main.pdf (11 pages, 271202 bytes).
```

## Figure and Citation Checks

- Expected figure slots in `main.tex`: 5
- Missing referenced PDF figures: 0
- BibTeX compilation: completed
- Missing citations after final compile: 0
- Internal `workbench/` paths in venue `main.tex`: none
- Internal `outputs/runs/` paths in venue `main.tex`: none
- `[RE...]` placeholders in venue `main.tex`: none

## ChkTeX

ChkTeX reported `31` style warnings. They are non-blocking for the current
arXiv-first package and mostly concern typography preferences, including:

- non-breaking spaces before citations,
- date hyphen style,
- false positives around `\texttt{...}` and labels.

No ChkTeX item indicates a failed compile, missing figure, missing citation, or
scientific-claim problem.

## Fixes Applied Before Final Compile

- The central research question was converted from an overlong display-math
  sentence to an emphasized quote.
- The long code-like feature name
  `clean_artifact_risk_raw_mean` was made breakable in the venue `main.tex`.
- PDF figures were regenerated from existing evidence tables.

## Claim Boundary

No experiments, training runs, raw data copies, thresholds, or scientific
interpretations were changed. This was a packaging and compile-readiness step
only.

## Decision

`proceed_to_submission_bundle_and_author_metadata_review`

The arXiv-first paper package is now locally compileable. The next step is to
prepare a clean submission bundle and fill or explicitly mark author metadata,
funding, acknowledgements, and venue-policy statements.
