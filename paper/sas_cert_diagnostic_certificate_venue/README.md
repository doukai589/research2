# arXiv-First Venue Package

This package is generated from `paper/sas_cert_diagnostic_certificate`.

## Current Route

Default route: `arxiv_article_first`.

This is the safest first packaging route because it preserves the current
article structure and claim boundary before committing to IEEE/ACM/Springer
format constraints.

## Figure Conversion

The generated `main.tex` expects PDF figures in `figures/`.

SVG originals are preserved in `figures_svg_original/`. Convert them with one of
the following commands when the corresponding tool is available:

```bash
inkscape figures_svg_original/figure1_certificate_overview.svg --export-type=pdf --export-filename=figures/figure1_certificate_overview.pdf
```

or:

```bash
rsvg-convert -f pdf -o figures/figure1_certificate_overview.pdf figures_svg_original/figure1_certificate_overview.svg
```

No conversion was performed during generation because no local SVG conversion
tool was detected.

## Claim Boundary

No new experiments or claims are introduced by this venue package. The paper
still supports SAS-Cert as a diagnostic reliability certificate and does not
promote current weighting/rejection policies as deployable methods.
