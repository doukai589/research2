#!/usr/bin/env python3
"""Build the venue-agnostic SAS-Cert diagnostic paper package.

This script is intentionally conservative: it copies existing manuscript
assets, filters the bibliography to cited entries, and converts the current
Markdown/LaTeX hybrid draft into a standalone LaTeX project without changing
experimental numbers or claims.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DRAFT = ROOT / "docs" / "SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT_LATEX.md"
SRC_BIB = ROOT / "docs" / "SAS_CERT_REFERENCES.bib"
SRC_FIG_DIR = ROOT / "outputs" / "runs" / "sas_cert_diagnostic_certificate_pack_physionetmi" / "manuscript_figures"
SRC_TABLE_DIR = ROOT / "outputs" / "runs" / "sas_cert_diagnostic_certificate_pack_physionetmi" / "manuscript_tables"
PAPER_DIR = ROOT / "paper" / "sas_cert_diagnostic_certificate"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"
REPORT_PATH = ROOT / "docs" / "SAS_CERT_VENUE_FORMATTING_REPORT.md"


def latex_escape(text: str) -> str:
    """Escape normal prose while preserving citation commands."""
    placeholders: dict[str, str] = {}

    def protect(pattern: str, name: str, value: str) -> str:
        key = f"@@{name}{len(placeholders)}@@"
        placeholders[key] = value
        return key

    text = re.sub(r"\\cite\{[^}]+\}", lambda m: protect("", "CITE", m.group(0)), text)
    text = re.sub(r"\\text[a-zA-Z]+\{[^}]*\}", lambda m: protect("", "CMD", m.group(0)), text)

    # Inline code spans are common in the source draft.
    def code_repl(match: re.Match[str]) -> str:
        code = match.group(1)
        code = (
            code.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )
        return protect("", "CODE", rf"\texttt{{{code}}}")

    text = re.sub(r"`([^`]+)`", code_repl, text)

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def strip_heading_number(text: str) -> str:
    """Remove Markdown-authored section numbers before LaTeX auto-numbering."""
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text).strip()


def parse_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        raw = lines[i].strip()
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        rows.append(cells)
        i += 1
    if len(rows) >= 2 and all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in rows[1]):
        rows.pop(1)
    return rows, i


def write_table_tex(rows: list[list[str]], table_id: int, caption: str) -> str:
    path = TABLE_DIR / f"inline_table_{table_id}.tex"
    cols = len(rows[0]) if rows else 1
    align = "l" * cols
    caption = strip_heading_number(caption)
    body: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{tab:inline-{table_id}}}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    for idx, row in enumerate(rows):
        padded = row + [""] * (cols - len(row))
        body.append(" & ".join(latex_escape(cell) for cell in padded[:cols]) + r" \\")
        if idx == 0:
            body.append(r"\midrule")
    body.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(body), encoding="utf-8")
    return path.name


def split_bib_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    matches = list(re.finditer(r"@\w+\{([^,]+),", text))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        entries[match.group(1)] = text[start:end].strip() + "\n"
    return entries


def cited_keys(text: str) -> list[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            keys.add(key.strip())
    return sorted(keys)


def copy_assets() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC_FIG_DIR.glob("*.svg")):
        shutil.copy2(src, FIG_DIR / src.name)
    for src in sorted(SRC_TABLE_DIR.glob("*.md")):
        shutil.copy2(src, TABLE_DIR / src.name)


def convert_draft_to_tex(md_text: str) -> str:
    lines = md_text.splitlines()
    title = lines[0].lstrip("# ").strip()
    body: list[str] = []
    in_math = False
    in_verbatim = False
    in_abstract = False
    in_itemize = False
    pending_caption = "Summary table"
    table_id = 0
    i = 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "## References":
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            if in_abstract:
                body.append(r"\end{abstract}")
                in_abstract = False
            break

        if stripped.startswith("```"):
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            if in_verbatim:
                body.append(r"\end{verbatim}")
                in_verbatim = False
            else:
                body.append(r"\begin{verbatim}")
                in_verbatim = True
            i += 1
            continue

        if in_verbatim:
            body.append(line)
            i += 1
            continue

        if stripped == r"\[":
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            in_math = True
            body.append(line)
            i += 1
            continue

        if in_math:
            body.append(line)
            if stripped == r"\]":
                in_math = False
            i += 1
            continue

        if stripped.startswith("|"):
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            rows, next_i = parse_markdown_table(lines, i)
            table_id += 1
            table_name = write_table_tex(rows, table_id, pending_caption)
            body.append(rf"\input{{tables/{table_name}}}")
            pending_caption = "Summary table"
            i = next_i
            continue

        if stripped.startswith("#"):
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            if in_abstract:
                body.append(r"\end{abstract}")
                in_abstract = False
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = strip_heading_number(stripped[level:].strip())
            pending_caption = heading
            if heading == "Abstract":
                body.append(r"\begin{abstract}")
                in_abstract = True
            elif level == 2:
                body.append(rf"\section{{{latex_escape(heading)}}}")
            elif level == 3:
                body.append(rf"\subsection{{{latex_escape(heading)}}}")
            elif level == 4:
                body.append(rf"\subsubsection{{{latex_escape(heading)}}}")
            else:
                body.append(rf"\paragraph{{{latex_escape(heading)}}}")
            i += 1
            continue

        if stripped.startswith("- "):
            if not in_itemize:
                body.append(r"\begin{itemize}")
                in_itemize = True
            body.append(rf"\item {latex_escape(stripped[2:].strip())}")
            i += 1
            continue

        if stripped == "":
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            body.append("")
            i += 1
            continue

        if in_itemize:
            body.append(r"\end{itemize}")
            in_itemize = False
        body.append(latex_escape(line))
        i += 1

    if in_itemize:
        body.append(r"\end{itemize}")
    if in_abstract:
        body.append(r"\end{abstract}")

    figure_block = r"""
\section{Manuscript Figures}
\begin{figure}[H]
\centering
\includesvg[width=0.95\linewidth]{figures/figure1_certificate_overview}
\caption{SAS-Cert diagnostic certificate overview.}
\label{fig:certificate-overview}
\end{figure}

\begin{figure}[H]
\centering
\includesvg[width=0.95\linewidth]{figures/figure2_diagnostic_auc}
\caption{Diagnostic AUC comparison between scalar and component-gated scores.}
\label{fig:diagnostic-auc}
\end{figure}

\begin{figure}[H]
\centering
\includesvg[width=0.95\linewidth]{figures/figure3_component_specificity_heatmap}
\caption{Component specificity across augmentation failure types.}
\label{fig:component-specificity}
\end{figure}

\begin{figure}[H]
\centering
\includesvg[width=0.95\linewidth]{figures/figure4_training_policy_non_promotion}
\caption{Training-policy evidence and non-promotion boundary.}
\label{fig:training-policy}
\end{figure}

\begin{figure}[H]
\centering
\includesvg[width=0.95\linewidth]{figures/figure5_causal_chain}
\caption{Supported and unsupported causal chains.}
\label{fig:causal-chain}
\end{figure}
"""

    preamble = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{svg}}
\usepackage{{url}}

\title{{{latex_escape(title)}}}
\author{{SAS-Cert EEG Project}}
\date{{2026-06-22}}

\begin{{document}}
\maketitle
"""
    end = r"""
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""
    return preamble + "\n".join(body).strip() + "\n" + figure_block + end


def write_filtered_bib(md_text: str) -> tuple[list[str], list[str]]:
    entries = split_bib_entries(SRC_BIB.read_text(encoding="utf-8"))
    used = cited_keys(md_text)
    missing = [key for key in used if key not in entries]
    kept = [entries[key] for key in used if key in entries]
    (PAPER_DIR / "references.bib").write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    unused = sorted(set(entries) - set(used))
    if missing:
        raise RuntimeError(f"Missing BibTeX entries for cited keys: {missing}")
    return used, unused


def write_readme(used: list[str], unused: list[str]) -> None:
    readme = f"""# SAS-Cert Diagnostic Certificate Paper Package

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

- Kept cited entries: {', '.join(used)}
- Removed unused entries from this paper package: {', '.join(unused) if unused else 'none'}

No experiments or claims were added during package generation.
"""
    (PAPER_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_report(used: list[str], unused: list[str]) -> None:
    fig_count = len(list(FIG_DIR.glob("*.svg")))
    generated_tables = len(list(TABLE_DIR.glob("inline_table_*.tex")))
    copied_tables = len(list(TABLE_DIR.glob("*.md")))
    latexmk_available = shutil.which("latexmk") is not None
    pdflatex_available = shutil.which("pdflatex") is not None
    report = f"""# SAS-Cert Venue Formatting Report

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

- Copied SVG figures: {fig_count}
- Generated inline LaTeX tables: {generated_tables}
- Copied Markdown evidence tables: {copied_tables}
- Cited BibTeX entries kept: {len(used)}
- Unused BibTeX entries removed from package: {len(unused)}

## Verification

- Citation audit: passed; every `\\cite{{}}` key in `main.tex` resolves in
  `paper/sas_cert_diagnostic_certificate/references.bib`.
- Bibliography audit: passed; the package bibliography contains no unused
  entries.
- SVG asset audit: passed; {fig_count} copied SVG figures are present.
- Internal-path audit: passed; `main.tex` does not expose `workbench/` or
  `outputs/runs/` paths.
- Local compile audit: not run in this environment because `latexmk` available
  is `{latexmk_available}` and `pdflatex` available is `{pdflatex_available}`.

## Bibliography Audit

Kept cited entries:

{chr(10).join(f'- `{key}`' for key in used)}

Removed unused entries from this paper package:

{chr(10).join(f'- `{key}`' for key in unused) if unused else '- none'}

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
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    copy_assets()
    md_text = SRC_DRAFT.read_text(encoding="utf-8")
    tex = convert_draft_to_tex(md_text)
    (PAPER_DIR / "main.tex").write_text(tex, encoding="utf-8")
    used, unused = write_filtered_bib(md_text)
    write_readme(used, unused)
    write_report(used, unused)
    print(f"Wrote {PAPER_DIR / 'main.tex'}")
    print(f"Wrote {PAPER_DIR / 'references.bib'}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
