# SAS-Cert Venue Formatting Report

## Status

Completed venue-agnostic LaTeX paper package generation.

## Inputs

- Draft: `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT_LATEX.md`
- Bibliography: `docs/SAS_CERT_REFERENCES.bib`
- Figures: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures`
- Tables: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables`

## Outputs

- Main LaTeX file: `paper/sas_cert_diagnostic_certificate/main.tex`
- Filtered bibliography: `paper/sas_cert_diagnostic_certificate/references.bib`
- Figures directory: `paper/sas_cert_diagnostic_certificate/figures`
- Tables directory: `paper/sas_cert_diagnostic_certificate/tables`
- Package README: `paper/sas_cert_diagnostic_certificate/README.md`

## Formatting Actions

- Converted the Markdown/LaTeX hybrid manuscript into standalone `article` LaTeX.
- Converted inline manuscript tables into generated LaTeX table inputs.
- Copied manuscript SVG figures into the paper package.
- Copied the original Markdown evidence tables into the paper package for traceability.
- Filtered BibTeX entries to cited keys only.

## Counts

- Copied SVG figures: 5
- Generated inline LaTeX tables: 4
- Copied Markdown evidence tables: 6
- Cited BibTeX entries kept: 10
- Unused BibTeX entries removed from package: 3

## Verification

- Citation audit: passed; every `\cite{}` key in `main.tex` resolves in
  `paper/sas_cert_diagnostic_certificate/references.bib`.
- Bibliography audit: passed; the package bibliography contains no unused
  entries.
- SVG asset audit: passed; 5 copied SVG figures are present.
- Internal-path audit: passed; `main.tex` does not expose `workbench/` or
  `outputs/runs/` paths.
- Local compile audit: not run in this environment because `latexmk` available
  is `False` and `pdflatex` available is `False`.

## Bibliography Audit

Kept cited entries:

- `Bollens2022SubjectInvariantVAE`
- `Ding2025RCC`
- `Ding2026JSCCRA`
- `Lee2025LargeBrainwaveFoundation`
- `Li2025FrequencyTemporalMI`
- `Sawangjai2022EEGANet`
- `Sirca2026LoRAEEG`
- `Wang2025CBraMod`
- `Winkler2011ICArtifact`
- `Yang2026STEEGFormer`

Removed unused entries from this paper package:

- `Heremans2022SleepAugDomainAdapt`
- `Liao2026EEGTune`
- `Liu2026MIRepNet`

## Claim Boundary

No new experiments were run and no new claims were introduced. The package
preserves the current claim boundary: SAS-Cert is supported as a diagnostic EEG
augmentation reliability certificate, while current weighting/rejection policies
are not promoted as deployable training methods.

## Compile Note

`main.tex` uses the LaTeX `svg` package. Build with a command such as:

```bash
latexmk -pdf -shell-escape main.tex
```
