#!/usr/bin/env python3
"""Build an arXiv-first venue package from the SAS-Cert paper package.

The package intentionally avoids running new experiments or changing claims.
It converts the LaTeX source to prefer PDF figures, while preserving the SVG
originals when local SVG conversion tools are unavailable.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "sas_cert_diagnostic_certificate"
DST = ROOT / "paper" / "sas_cert_diagnostic_certificate_venue"
REPORT = ROOT / "docs" / "SAS_CERT_VENUE_SELECTION_REPORT.md"


def ensure_dirs() -> None:
    (DST / "figures").mkdir(parents=True, exist_ok=True)
    (DST / "figures_svg_original").mkdir(parents=True, exist_ok=True)
    (DST / "tables").mkdir(parents=True, exist_ok=True)


def copy_static_assets() -> None:
    shutil.copy2(SRC / "references.bib", DST / "references.bib")
    for path in sorted((SRC / "tables").glob("*")):
        if path.is_file():
            shutil.copy2(path, DST / "tables" / path.name)
    for path in sorted((SRC / "figures").glob("*.svg")):
        shutil.copy2(path, DST / "figures_svg_original" / path.name)


def convert_main_for_arxiv(tex: str) -> str:
    tex = tex.replace(r"\usepackage{svg}" + "\n", "")
    tex = tex.replace(
        r"""\[
\text{How can we decide whether } x' \text{ is a beneficial subject-style variation}
\quad \text{or a harmful content/physiology/artifact distortion?}
\]""",
        r"""\begin{quote}
\emph{How can we decide whether an augmented EEG sample is a beneficial
subject-style variation or a harmful content/physiology/artifact distortion?}
\end{quote}""",
    )
    tex = tex.replace(
        r"\texttt{clean\_artifact\_risk\_raw\_mean}",
        r"\texttt{clean\_\allowbreak artifact\_\allowbreak risk\_\allowbreak raw\_\allowbreak mean}",
    )

    def repl(match: re.Match[str]) -> str:
        width = match.group(1)
        fig_stem = Path(match.group(2)).name
        pdf_path = f"figures/{fig_stem}.pdf"
        placeholder = (
            rf"\IfFileExists{{{pdf_path}}}"
            rf"{{\includegraphics[width={width}]{{{pdf_path}}}}}"
            rf"{{\fbox{{\parbox[c][0.24\textheight][c]{{0.88\linewidth}}"
            rf"{{\centering Missing converted figure: {fig_stem}.pdf\\"
            rf"Original SVG is in figures\_svg\_original/.}}}}}}"
        )
        return placeholder

    return re.sub(r"\\includesvg\[width=([^\]]+)\]\{([^}]+)\}", repl, tex)


def write_notes() -> None:
    notes = """# arXiv-First Venue Package

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
"""
    (DST / "README.md").write_text(notes, encoding="utf-8")


def build() -> None:
    ensure_dirs()
    copy_static_assets()
    tex = (SRC / "main.tex").read_text(encoding="utf-8")
    (DST / "main.tex").write_text(convert_main_for_arxiv(tex), encoding="utf-8")
    write_notes()


if __name__ == "__main__":
    build()
    print(f"Wrote {DST}")
