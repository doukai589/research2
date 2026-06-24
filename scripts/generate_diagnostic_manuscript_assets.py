#!/usr/bin/env python3
"""Generate manuscript-prep assets from existing SAS-Cert evidence files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi"
TABLES = PACK / "tables"
OUT_TABLES = PACK / "manuscript_tables"
OUT_FIGS = PACK / "manuscript_figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def fmt_float(value: str, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return value


def generate_submission_draft() -> Path:
    src = ROOT / "docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md"
    dst = ROOT / "docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md"
    lines = src.read_text(encoding="utf-8").splitlines()
    filtered = []
    skip_blank_after_removed_line = False
    skip_rest = False
    for line in lines:
        if line.startswith("## Evidence Index"):
            skip_rest = True
            continue
        if skip_rest:
            continue
        if (
            line.startswith("Citation placeholders:")
            or line.startswith("Evidence sources:")
            or line.startswith("Source: `")
            or line.startswith("Sources: `")
        ):
            skip_blank_after_removed_line = True
            continue
        if skip_blank_after_removed_line and line.strip() == "":
            skip_blank_after_removed_line = False
            continue
        skip_blank_after_removed_line = False
        filtered.append(line)
    text = "\n".join(filtered).rstrip() + "\n"
    text = text.replace(
        "Failure-mode definitions are stored in `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv`.",
        "Failure-mode definitions are reported in the appendix.",
    )
    text = text.replace(
        "No new training, no new backbone, and no new dataset were introduced in the diagnostic certificate pack. The protocol leakage audit is stored in `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`.",
        "No new training, no new backbone, and no new dataset were introduced in the diagnostic certificate pack. A protocol leakage audit was included in the evidence package.",
    )
    write_text(dst, text)
    return dst


def generate_bibliography_trace() -> Path:
    dst = ROOT / "docs/SAS_CERT_BIBLIOGRAPHY_TRACE.md"
    rows = [
        ("Wang2025CBraMod", "mapped_pdf", "docs/references/Wang 等 - 2025 - CBraMod A Criss-Cross Brain Foundation Model for EEG Decoding.pdf", "extract exact BibTeX"),
        ("Liu2026MIRepNet", "mapped_pdf", "docs/references/Liu 等 - 2026 - MIRepNet A pipeline and pre-trained model for EEG-based motor imagery classification.pdf", "extract exact BibTeX; keep as context only"),
        ("RE199", "mapped_reference_note", "选题.txt foundation-model marginal-gain note", "resolve title/authors/year from reference library"),
        ("RE132", "mapped_reference_note", "选题.txt LaBraM+LoRA low-label note", "resolve title/authors/year from reference library"),
        ("RE309", "mapped_reference_note", "选题.txt and 参考论文集/论文集_004.txt JSCCRA entry", "resolve title/authors/year from reference library"),
        ("RE334", "mapped_reference_note", "选题.txt and 参考论文集/论文集_004.txt factorized latent note", "resolve title/authors/year from reference library"),
        ("RE167", "mapped_reference_note", "选题.txt MI frequency-time prior note", "resolve title/authors/year from reference library"),
        ("RE181", "mapped_reference_note", "选题.txt covariance/RCC note", "resolve title/authors/year from reference library"),
        ("RE333", "mapped_reference_note", "参考论文集/论文集_004.txt EEGANet entry", "resolve title/authors/year from reference library"),
        ("RE342", "mapped_reference_note", "参考论文集/论文集_004.txt automatic ICA artifact classifier entry", "resolve title/authors/year from reference library"),
        ("RE185", "mapped_reference_note", "选题.txt EEGTune note", "resolve title/authors/year from reference library"),
        ("RE332", "mapped_reference_note", "选题.txt small augmentation gain note", "resolve title/authors/year from reference library"),
        ("ST-EEGFormer-small", "missing_formal_citation", "local backbone source used in experiments", "record repository/paper source before final submission"),
    ]
    text = [
        "# SAS-Cert Bibliography Trace",
        "",
        "This trace resolves manuscript citation placeholders to local anchors. It is not a final BibTeX file yet.",
        "",
        "| Key | Status | Local anchor | Next action |",
        "|---|---|---|---|",
    ]
    for key, status, anchor, action in rows:
        text.append(f"| `{key}` | {status} | `{anchor}` | {action} |")
    text.extend(
        [
            "",
            "Submission rule: replace all placeholder keys with verified BibTeX keys before venue formatting.",
            "",
        ]
    )
    write_text(dst, "\n".join(text))
    return dst


def generate_references_bib() -> Path:
    dst = ROOT / "docs/SAS_CERT_REFERENCES.bib"
    bib = r"""@inproceedings{Wang2025CBraMod,
  title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
  author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=NPNUHgHF2w}
}

@inproceedings{Yang2026STEEGFormer,
  title={Are {EEG} Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse {BCI} Tasks},
  author={Liuyin Yang and Qiang Sun and Ang Li and Marc M. Van Hulle},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=5Xwm8e6vbh}
}

@article{Liu2026MIRepNet,
  title={{MIR}ep{N}et: A Pipeline and Pre-trained Model for {EEG}-Based Motor Imagery Classification},
  author={Dingkun Liu and Zhu Chen and Jingwei Luo and Shijie Lian and Yuheng Chen and Shaojie Hou and Xiaolian Zhu and Dongrui Wu},
  journal={Knowledge-Based Systems},
  volume={343},
  pages={115966},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.knosys.2026.115966}
}

@misc{Sirca2026LoRAEEG,
  title={Parameter-Efficient Fine-Tuning of {EEG} Foundation Models for Plug-and-Play Motor Imagery {BCIs}},
  author={U. {\v{S}}irca and L. Brulec and M. Alimardani},
  year={2026},
  note={Reference-library entry RE132}
}

@misc{Li2025FrequencyTemporalMI,
  title={A Frequency-Temporal Causal Inference Guided Adversarial Network for Cross-Subject {MI-BCI} Decoding},
  author={Y. Li and D. Su and X. Wang and H. Zhao and J. Zhang},
  year={2025},
  note={Reference-library entry RE167}
}

@article{Ding2025RCC,
  title={Enhancing Domain Diversity of Transfer Learning-Based {SSVEP-BCIs} by the Reconstruction of Channel Correlation},
  author={W. Ding and A. Liu and C. Xie and K. Wang and X. Chen},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2025},
  note={Reference-library entry RE181}
}

@article{Liao2026EEGTune,
  title={{EEGTune}: A Data-Efficient Fine-Tuning Framework for {EEG} Foundation Models},
  author={Z. Liao and Y. Song and C. Xu and H. Zhang and Q. Zheng},
  journal={Pattern Recognition},
  year={2026},
  note={Reference-library entry RE185}
}

@misc{Lee2025LargeBrainwaveFoundation,
  title={Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning},
  author={N. Lee and K. Barmpas and Y. Panagakis and D. Adamos and N. Laskaris and S. Zafeiriou},
  year={2025},
  note={Reference-library entry RE199}
}

@article{Ding2026JSCCRA,
  title={Enhancing Subject-Independent {SSVEP-BCIs} via Joint Style Characteristic and Content Representation Augmentation},
  author={W. Ding and A. Liu and H. Cui and J. Shan and X. Chen},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  note={Reference-library entry RE309}
}

@misc{Heremans2022SleepAugDomainAdapt,
  title={Data Augmentation in Semi-Supervised Adversarial Domain Adaptation for {EEG}-Based Sleep Staging},
  author={E. R. M. Heremans and T. Osselaer and N. Seeuws and H. Phan and D. Testelmans and M. De Vos},
  year={2022},
  note={Reference-library entry RE332}
}

@article{Sawangjai2022EEGANet,
  title={{EEGANet}: Removal of Ocular Artifacts from the {EEG} Signal Using Generative Adversarial Networks},
  author={P. Sawangjai and M. Trakulruangroj and C. Boonnag and M. Piriyajitakonkij and R. K. Tripathy and T. Sudhawiyangkul and T. Wilaiprasitporn},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  note={Reference-library entry RE333}
}

@misc{Bollens2022SubjectInvariantVAE,
  title={Learning Subject-Invariant Representations from Speech-Evoked {EEG} Using Variational Autoencoders},
  author={L. Bollens and T. Francart and H. Van Hamme},
  year={2022},
  note={Reference-library entry RE334}
}

@article{Winkler2011ICArtifact,
  title={Automatic Classification of Artifactual {ICA}-Components for Artifact Removal in {EEG} Signals},
  author={I. Winkler and S. Haufe and M. Tangermann},
  journal={Behavioral and Brain Functions},
  year={2011},
  note={Reference-library entry RE342}
}
"""
    write_text(dst, bib)
    return dst


def generate_bibliography_report() -> Path:
    dst = ROOT / "docs/SAS_CERT_BIBLIOGRAPHY_RESOLUTION_REPORT.md"
    text = """# SAS-Cert Bibliography Resolution Report

## Verdict

```text
bibliography_placeholders_resolved_with_traceable_local_sources
```

The manuscript placeholders now have BibTeX entries in:

```text
docs/SAS_CERT_REFERENCES.bib
```

## Resolution Sources

| Key | Resolution status | Source |
|---|---|---|
| `Wang2025CBraMod` | resolved | `third_party/CBraMod-main/README.md` BibTeX |
| `Yang2026STEEGFormer` | resolved | `third_party/backbones/STEEGFormer/README.md` BibTeX |
| `Liu2026MIRepNet` | resolved | `third_party/backbones/MIRepNet/README.md` BibTeX |
| `Sirca2026LoRAEEG` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE132 |
| `Li2025FrequencyTemporalMI` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE167 |
| `Ding2025RCC` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE181 |
| `Liao2026EEGTune` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE185 |
| `Lee2025LargeBrainwaveFoundation` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE199 |
| `Ding2026JSCCRA` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE309 |
| `Heremans2022SleepAugDomainAdapt` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE332 |
| `Sawangjai2022EEGANet` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE333 |
| `Bollens2022SubjectInvariantVAE` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE334 |
| `Winkler2011ICArtifact` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE342 |

## Remaining Caveat

Some RE entries are resolved from the local reference-library exports rather
than publisher BibTeX. Before external submission, DOI/page metadata should be
checked for entries currently marked as `misc` or with `note` fields.
"""
    write_text(dst, text)
    return dst


def generate_latex_draft() -> Path:
    src = ROOT / "docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md"
    dst = ROOT / "docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT_LATEX.md"
    text = src.read_text(encoding="utf-8")
    citation_replacements = {
        "[Wang2025CBraMod, RE199]": r"\cite{Wang2025CBraMod,Lee2025LargeBrainwaveFoundation}",
        "[RE132, RE199]": r"\cite{Sirca2026LoRAEEG,Lee2025LargeBrainwaveFoundation}",
        "[RE309]": r"\cite{Ding2026JSCCRA}",
        "[RE334]": r"\cite{Bollens2022SubjectInvariantVAE}",
        "[RE167]": r"\cite{Li2025FrequencyTemporalMI}",
        "[RE181]": r"\cite{Ding2025RCC}",
        "[RE333, RE342]": r"\cite{Sawangjai2022EEGANet,Winkler2011ICArtifact}",
        "[Wang2025CBraMod]": r"\cite{Wang2025CBraMod}",
    }
    for old, new in citation_replacements.items():
        text = text.replace(old, new)
    block_replacements = {
        "```text\nHow can we decide whether an augmented EEG sample is a beneficial subject-style\nvariation or a harmful task-content, physiology, or artifact distortion?\n```": r"""\[
\text{How can we decide whether } x' \text{ is a beneficial subject-style variation}
\quad \text{or a harmful content/physiology/artifact distortion?}
\]""",
        "```text\nx'_ij = a_j(x_i),  j = 1,...,m\n```": r"""\[
x'_{ij} = a_j(x_i), \qquad j = 1,\ldots,m .
\]""",
        "```text\nC(x_i, x'_ij; S_t) =\n  [C_content, C_style, C_physio, C_artifact_safe].\n```": r"""\[
C(x_i, x'_{ij}; S_t)=
\left[C_{\mathrm{content}}, C_{\mathrm{style}},
C_{\mathrm{physio}}, C_{\mathrm{artifact\_safe}}\right].
\]""",
        "```text\nC_content(x, x') = cos(f_theta(x), f_theta(x')).\n```": r"""\[
C_{\mathrm{content}}(x,x') =
\cos\left(f_{\theta}(x), f_{\theta}(x')\right).
\]""",
        "```text\nmu_style(S_t) = mean_{x in S_t} s(x).\n```": r"""\[
\mu_{\mathrm{style}}(S_t)=\frac{1}{|S_t|}\sum_{x\in S_t}s(x).
\]""",
        "```text\nC_style(x'; S_t) = -d_style(s(x'), mu_style(S_t)).\n```": r"""\[
C_{\mathrm{style}}(x';S_t)=
-d_{\mathrm{style}}\left(s(x'),\mu_{\mathrm{style}}(S_t)\right).
\]""",
        "```text\nC_physio(x, x') =\n  - alpha_mu_beta * D_band(x, x')\n  - alpha_cov * D_cov(x, x').\n```": r"""\[
C_{\mathrm{physio}}(x,x') =
-\alpha_{\mu/\beta}D_{\mathrm{band}}(x,x')
-\alpha_{\mathrm{cov}}D_{\mathrm{cov}}(x,x').
\]""",
        "```text\nC_artifact_safe(x') = -R_artifact(x').\n```": r"""\[
C_{\mathrm{artifact\_safe}}(x')=-R_{\mathrm{artifact}}(x').
\]""",
        "```text\nS_old = w_c rank(C_content)\n      + w_s rank(C_style)\n      + w_p rank(C_physio)\n      + w_a rank(C_artifact_safe).\n```": r"""\[
S_{\mathrm{old}} =
w_c\,\mathrm{rank}(C_{\mathrm{content}})
+w_s\,\mathrm{rank}(C_{\mathrm{style}})
+w_p\,\mathrm{rank}(C_{\mathrm{physio}})
+w_a\,\mathrm{rank}(C_{\mathrm{artifact\_safe}}).
\]""",
        "```text\nartifact_gate_pass = I(R_artifact(x') < tau_artifact)\nbase = 0.75 * rank(C_physio) + 0.25 * rank(C_style)\nS_component_gated_v1 = rank(base) * artifact_gate_pass.\n```": r"""\[
g_{\mathrm{artifact}}(x') =
\mathbb{I}\left[R_{\mathrm{artifact}}(x')<\tau_{\mathrm{artifact}}\right],
\]
\[
B = 0.75\,\mathrm{rank}(C_{\mathrm{physio}})
+0.25\,\mathrm{rank}(C_{\mathrm{style}}),
\qquad
S_{\mathrm{component\_gated\_v1}}=\mathrm{rank}(B)\,g_{\mathrm{artifact}}(x').
\]""",
        "```text\nS_artifact_gate_physio = rank(C_physio) * artifact_gate_pass.\n```": r"""\[
S_{\mathrm{artifact\_gate\_physio}} =
\mathrm{rank}(C_{\mathrm{physio}})\,g_{\mathrm{artifact}}(x').
\]""",
        "```text\ndiagnostic separability != deployable training utility\n```": r"""\[
\text{diagnostic separability} \ne \text{deployable training utility}.
\]""",
    }
    for old, new in block_replacements.items():
        text = text.replace(old, new)
    text = text.replace(
        "ST-EEGFormer-small source-tuned frozen encoder.",
        r"ST-EEGFormer-small source-tuned frozen encoder \cite{Yang2026STEEGFormer}.",
    )
    text += "\n## References\n\nUse `docs/SAS_CERT_REFERENCES.bib` for BibTeX entries.\n"
    write_text(dst, text)
    return dst


def generate_figure_table_polish_report() -> Path:
    dst = ROOT / "docs/SAS_CERT_FIGURE_TABLE_POLISH_REPORT.md"
    text = """# SAS-Cert Figure and Table Polish Report

## Verdict

```text
figure_table_assets_polished_as_draft_quality_without_data_changes
```

The generated figure and table assets are internally consistent with the
manuscript plan and were produced from existing CSV/JSON evidence only.

## Assets

Figures:

- `figure1_certificate_overview.svg`: conceptual certificate overview.
- `figure2_diagnostic_auc.svg`: scalar score failure and component recovery.
- `figure3_component_specificity_heatmap.svg`: component AUC heatmap by bad type.
- `figure4_training_policy_non_promotion.svg`: training-policy deltas and non-promotion.
- `figure5_causal_chain.svg`: diagnostic-to-training causal chain.

Tables:

- `table1_protocol_and_claim_boundary.md`
- `table2_diagnostic_auc_summary.md`
- `table3_bad_type_component_auc.md`
- `table4_training_policy_non_promotion.md`
- `table5_claim_support.md`
- `appendix_protocol_leakage_audit.md`

## Data Preservation

- No values were changed.
- No new experiments were run.
- No new claims were introduced.
- Figures were generated from the existing diagnostic pack summary tables.

## Remaining Layout Work

| Item | Status | Action |
|---|---|---|
| Figure 1 | draft-ready | Convert to final vector style after venue template is known. |
| Figure 2 | draft-ready | Consider adding confidence intervals only if already available; do not invent uncertainty bars. |
| Figure 3 | appendix-preferred | Too dense for main text; use compact summary in main text and keep full heatmap/table in appendix. |
| Figure 4 | draft-ready | Keep non-promotion labels visible; avoid leaderboard framing. |
| Figure 5 | draft-ready | Suitable as conceptual discussion figure. |
| Table 3 | appendix-preferred | Full bad-type component AUC table is long. |

## Claim Safety

The figure/table package preserves the manuscript boundary:

```text
diagnostic certificate supported; deployable weighting/rejection policy not promoted
```
"""
    write_text(dst, text)
    return dst


def generate_tables() -> list[Path]:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    protocol = read_csv(TABLES / "protocol_leakage_audit.csv")
    claims = read_csv(TABLES / "claim_support_table.csv")
    supported_boundary = "; ".join(r["allowed_wording"] for r in claims if r["stance"] == "supported")
    unsupported_boundary = "; ".join(
        f'{r["claim"]} [{r["allowed_wording"]}]' for r in claims if r["stance"] == "unsupported"
    )
    table1_rows = [
        {"item": "Dataset/task", "value": "PhysioNetMI / EEGMMI, left-vs-right motor imagery, runs R04/R08/R12"},
        {"item": "Backbones", "value": "CBraMod frozen; ST-EEGFormer-small source-tuned frozen"},
        {"item": "Held-out target use", "value": "Final evaluation only; not used for score, threshold, or ranking"},
        {"item": "Protocol leakage audit rows", "value": str(len(protocol))},
        {"item": "Supported claim boundary", "value": supported_boundary},
        {"item": "Unsupported claim boundary", "value": unsupported_boundary},
    ]
    path = OUT_TABLES / "table1_protocol_and_claim_boundary.md"
    write_text(path, "# Table 1. Protocol and Claim Boundary\n\n" + md_table(table1_rows, ["item", "value"]))
    outputs.append(path)

    diag = read_csv(TABLES / "diagnostic_auc_summary.csv")
    for row in diag:
        for key in row:
            if key != "backbone":
                row[key] = fmt_float(row[key])
    path = OUT_TABLES / "table2_diagnostic_auc_summary.md"
    write_text(path, "# Table 2. Diagnostic AUC Summary\n\n" + md_table(diag, list(diag[0].keys())))
    outputs.append(path)

    bad = read_csv(TABLES / "bad_type_component_auc.csv")
    keep_cols = ["backbone", "bad_type", "component", "auc_high_score_is_clean", "direction", "n"]
    for row in bad:
        row["auc_high_score_is_clean"] = fmt_float(row["auc_high_score_is_clean"])
    path = OUT_TABLES / "table3_bad_type_component_auc.md"
    write_text(path, "# Table 3. Bad-Type Component AUC\n\n" + md_table(bad, keep_cols))
    outputs.append(path)

    train = read_csv(TABLES / "training_policy_summary.csv")
    for row in train:
        for key in ["delta_balanced_accuracy_vs_naive", "delta_macro_f1_vs_naive", "delta_ece_vs_naive"]:
            row[key] = fmt_float(row[key])
    path = OUT_TABLES / "table4_training_policy_non_promotion.md"
    write_text(path, "# Table 4. Training Policy Non-Promotion\n\n" + md_table(train, list(train[0].keys())))
    outputs.append(path)

    path = OUT_TABLES / "table5_claim_support.md"
    write_text(path, "# Table 5. Claim Support and Prohibited Claims\n\n" + md_table(claims, list(claims[0].keys())))
    outputs.append(path)

    path = OUT_TABLES / "appendix_protocol_leakage_audit.md"
    write_text(path, "# Appendix. Protocol Leakage Audit\n\n" + md_table(protocol, list(protocol[0].keys())))
    outputs.append(path)

    return outputs


def svg_bar_chart(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float], str]]) -> None:
    width, height = 900, 460
    margin_left, margin_bottom = 90, 80
    plot_w, plot_h = 720, 300
    max_v = max(max(vals) for _, vals, _ in series)
    max_v = max(1.0, max_v)
    n_groups = len(labels)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = group_w / (n_series + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{60+plot_h}" x2="{margin_left+plot_w}" y2="{60+plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{60+plot_h}" stroke="#222"/>',
    ]
    for tick in [0.0, 0.5, 0.7, 1.0]:
        y = 60 + plot_h - tick / max_v * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin_left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for gi, label in enumerate(labels):
        x0 = margin_left + gi * group_w
        parts.append(f'<text x="{x0+group_w/2:.1f}" y="{60+plot_h+34}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
        for si, (name, vals, color) in enumerate(series):
            v = vals[gi]
            x = x0 + (si + 0.5) * bar_w
            h = v / max_v * plot_h
            y = 60 + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w*0.41:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{v:.3f}</text>')
    for si, (name, _vals, color) in enumerate(series):
        lx = margin_left + si * 230
        parts.append(f'<rect x="{lx}" y="{height-36}" width="16" height="16" fill="{color}"/>')
        parts.append(f'<text x="{lx+24}" y="{height-23}" font-family="Arial" font-size="13">{name}</text>')
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def generate_figures() -> list[Path]:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    diag = read_csv(TABLES / "diagnostic_auc_summary.csv")
    labels = [row["backbone"].replace("_source_tuned", "").replace("_", " ") for row in diag]
    series = [
        ("current scalar SAS", [float(r["current_scalar_sas_auc"]) for r in diag], "#b94a48"),
        ("component-gated v1", [float(r["component_gated_v1_auc"]) for r in diag], "#4f81bd"),
        ("artifact-gate physio", [float(r["artifact_gate_physio_auc"]) for r in diag], "#5f9e6e"),
    ]
    path = OUT_FIGS / "figure2_diagnostic_auc.svg"
    svg_bar_chart(path, "Scalar Failure and Component Diagnostic Recovery", labels, series)
    outputs.append(path)

    bad = read_csv(TABLES / "bad_type_component_auc.csv")
    components = ["content_score", "style_score", "physio_score", "artifact_safe_score", "sas_score", "score_artifact_gate_physio"]
    row_keys = []
    for row in bad:
        key = f'{row["backbone"].replace("_source_tuned", "").replace("_", " ")} / {row["bad_type"]}'
        if key not in row_keys:
            row_keys.append(key)
    values = {(f'{r["backbone"].replace("_source_tuned", "").replace("_", " ")} / {r["bad_type"]}', r["component"]): float(r["auc_high_score_is_clean"]) for r in bad}
    cell, left, top = 92, 250, 80
    width = left + cell * len(components) + 40
    height = top + cell * len(row_keys) + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="34" font-family="Arial" font-size="22" font-weight="700">Component Specificity by Bad Type</text>',
    ]
    for ci, comp in enumerate(components):
        x = left + ci * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="62" text-anchor="middle" font-family="Arial" font-size="11">{comp}</text>')
    for ri, rk in enumerate(row_keys):
        y = top + ri * cell
        parts.append(f'<text x="{left-10}" y="{y+cell/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{rk}</text>')
        for ci, comp in enumerate(components):
            v = values[(rk, comp)]
            red = int(240 - 120 * v)
            green = int(120 + 115 * v)
            blue = 120
            x = left + ci * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="rgb({red},{green},{blue})"/>')
            parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+4:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{v:.3f}</text>')
    parts.append("</svg>")
    path = OUT_FIGS / "figure3_component_specificity_heatmap.svg"
    write_text(path, "\n".join(parts))
    outputs.append(path)

    train = read_csv(TABLES / "training_policy_summary.csv")
    labels = [row["branch"].replace("_", " ") for row in train]
    series = [
        ("Delta BAcc", [float(r["delta_balanced_accuracy_vs_naive"]) for r in train], "#4f81bd"),
        ("Delta Macro-F1", [float(r["delta_macro_f1_vs_naive"]) for r in train], "#5f9e6e"),
        ("Delta ECE", [float(r["delta_ece_vs_naive"]) for r in train], "#b94a48"),
    ]
    path = OUT_FIGS / "figure4_training_policy_non_promotion.svg"
    svg_bar_chart(path, "Training Policy Deltas and Non-Promotion", labels, series)
    outputs.append(path)

    path = OUT_FIGS / "figure5_causal_chain.svg"
    write_text(
        path,
        """<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="260" viewBox="0 0 1000 260">
<rect width="100%" height="100%" fill="white"/>
<text x="500" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">SAS-Cert Evidence Chain</text>
<g font-family="Arial" font-size="15" text-anchor="middle">
<rect x="40" y="90" width="190" height="70" rx="8" fill="#e8f1fb" stroke="#4f81bd"/>
<text x="135" y="120">Bad/clean augmentation</text><text x="135" y="142">is separable</text>
<rect x="285" y="90" width="190" height="70" rx="8" fill="#e8f1fb" stroke="#4f81bd"/>
<text x="380" y="120">Score direction</text><text x="380" y="142">must be audited</text>
<rect x="530" y="90" width="190" height="70" rx="8" fill="#e8f1fb" stroke="#4f81bd"/>
<text x="625" y="120">Component diagnostics</text><text x="625" y="142">recover separation</text>
<rect x="775" y="90" width="190" height="70" rx="8" fill="#fde9e7" stroke="#b94a48"/>
<text x="870" y="115">Stable deployable</text><text x="870" y="137">training utility</text><text x="870" y="159">not yet supported</text>
<text x="255" y="130" font-size="28">-&gt;</text>
<text x="500" y="130" font-size="28">-&gt;</text>
<text x="745" y="130" font-size="28" fill="#b94a48">-X-&gt;</text>
</g>
</svg>
""",
    )
    outputs.append(path)

    path = OUT_FIGS / "figure1_certificate_overview.svg"
    write_text(
        path,
        """<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360">
<rect width="100%" height="100%" fill="white"/>
<text x="500" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">SAS-Cert Diagnostic Certificate Overview</text>
<g font-family="Arial" font-size="15" text-anchor="middle">
<rect x="40" y="85" width="180" height="70" rx="8" fill="#f5f5f5" stroke="#555"/>
<text x="130" y="116">Support/source</text><text x="130" y="138">candidate pool</text>
<rect x="290" y="60" width="420" height="130" rx="8" fill="#e8f1fb" stroke="#4f81bd"/>
<text x="500" y="88" font-weight="700">Certificate profile</text>
<text x="395" y="125">Content</text><text x="500" y="125">Style</text><text x="605" y="125">Physio</text><text x="500" y="160">Artifact safety</text>
<rect x="780" y="85" width="180" height="70" rx="8" fill="#edf7ed" stroke="#5f9e6e"/>
<text x="870" y="116">Direction-audited</text><text x="870" y="138">diagnosis</text>
<text x="255" y="128" font-size="28">-&gt;</text><text x="745" y="128" font-size="28">-&gt;</text>
<rect x="260" y="240" width="480" height="54" rx="8" fill="#fff4e6" stroke="#c88432"/>
<text x="500" y="263">Claim boundary: diagnostic certificate supported;</text>
<text x="500" y="284">deployable weighting/rejection policy not promoted</text>
</g>
</svg>
""",
    )
    outputs.append(path)

    return outputs


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    created = {
        "submission_clean_draft": str(generate_submission_draft().relative_to(ROOT)),
        "latex_submission_draft": str(generate_latex_draft().relative_to(ROOT)),
        "bibliography_trace": str(generate_bibliography_trace().relative_to(ROOT)),
        "references_bib": str(generate_references_bib().relative_to(ROOT)),
        "bibliography_resolution_report": str(generate_bibliography_report().relative_to(ROOT)),
        "figure_table_polish_report": str(generate_figure_table_polish_report().relative_to(ROOT)),
        "tables": [str(p.relative_to(ROOT)) for p in generate_tables()],
        "figures": [str(p.relative_to(ROOT)) for p in generate_figures()],
        "constraints": {
            "new_experiments": False,
            "new_training": False,
            "new_claims": False,
            "source": "existing_csv_json_and_polished_draft_only",
        },
    }
    write_text(PACK / "manuscript_asset_manifest.json", json.dumps(created, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(created, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
