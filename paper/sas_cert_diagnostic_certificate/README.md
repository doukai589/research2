# SAS-Cert Diagnostic Certificate Paper Package

This directory is a venue-agnostic LaTeX package generated from the current
diagnostic manuscript assets.

## Build

Use a LaTeX engine with SVG support, for example:

```bash
latexmk -pdf -shell-escape main.tex
```

The `-shell-escape` flag is required by the LaTeX `svg` package when converting
SVG figures during compilation.

## Contents

- `main.tex`: standalone manuscript source.
- `references.bib`: cited BibTeX entries only.
- `figures/`: copied SVG manuscript figures.
- `tables/`: generated inline LaTeX tables plus copied Markdown evidence tables.

## Bibliography Filtering

- Kept cited entries: Bollens2022SubjectInvariantVAE, Ding2025RCC, Ding2026JSCCRA, Lee2025LargeBrainwaveFoundation, Li2025FrequencyTemporalMI, Sawangjai2022EEGANet, Sirca2026LoRAEEG, Wang2025CBraMod, Winkler2011ICArtifact, Yang2026STEEGFormer
- Removed unused entries from this paper package: Heremans2022SleepAugDomainAdapt, Liao2026EEGTune, Liu2026MIRepNet

No experiments or claims were added during package generation.
