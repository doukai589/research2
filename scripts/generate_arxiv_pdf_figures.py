#!/usr/bin/env python3
"""Regenerate arXiv-compatible PDF figures from existing SAS-Cert evidence.

This script reads only the frozen diagnostic evidence tables and writes PDF
figures expected by `paper/sas_cert_diagnostic_certificate_venue/main.tex`.
It does not run experiments, change metrics, or introduce new claims.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "runs" / "sas_cert_diagnostic_certificate_pack_physionetmi" / "tables"
OUT = ROOT / "paper" / "sas_cert_diagnostic_certificate_venue" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def clean_backbone(name: str) -> str:
    return name.replace("_source_tuned", "").replace("_", " ")


def fig1_certificate_overview() -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("SAS-Cert Diagnostic Certificate Overview", fontsize=15, weight="bold", pad=12)

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    box(0.05, 0.55, 0.18, 0.18, "Support/source\ncandidate pool", "#f4f4f4", "#666666")
    box(0.34, 0.48, 0.32, 0.32, "Certificate profile\n\nContent | Style\nPhysiology | Artifact safety", "#e9f2fb", "#4f81bd")
    box(0.78, 0.55, 0.17, 0.18, "Direction-audited\ndiagnosis", "#ecf6ec", "#5f9e6e")
    box(
        0.22,
        0.13,
        0.56,
        0.14,
        "Claim boundary: diagnostic certificate supported;\nweighting/rejection policy not promoted",
        "#fff3df",
        "#c88432",
    )
    for x0, x1 in [(0.24, 0.34), (0.66, 0.78)]:
        ax.add_patch(FancyArrowPatch((x0, 0.64), (x1, 0.64), arrowstyle="->", mutation_scale=18, linewidth=1.4))
    return save(fig, "figure1_certificate_overview.pdf")


def fig2_diagnostic_auc() -> Path:
    rows = read_csv(TABLES / "diagnostic_auc_summary.csv")
    labels = [clean_backbone(r["backbone"]) for r in rows]
    series = [
        ("Current scalar SAS", [float(r["current_scalar_sas_auc"]) for r in rows], "#b94a48"),
        ("Component-gated v1", [float(r["component_gated_v1_auc"]) for r in rows], "#4f81bd"),
        ("Artifact-gate physio", [float(r["artifact_gate_physio_auc"]) for r in rows], "#5f9e6e"),
    ]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for idx, (name, vals, color) in enumerate(series):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width=width, label=name, color=color)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=8, padding=2)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, label="chance AUC")
    ax.axhline(0.7, color="#999999", linestyle=":", linewidth=1, label="diagnostic target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clean-vs-bad AUC")
    ax.set_title("Scalar Failure and Component Diagnostic Recovery")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return save(fig, "figure2_diagnostic_auc.pdf")


def fig3_component_specificity_heatmap() -> Path:
    rows = read_csv(TABLES / "bad_type_component_auc.csv")
    components = [
        "content_score",
        "style_score",
        "physio_score",
        "artifact_safe_score",
        "sas_score",
        "score_artifact_gate_physio",
    ]
    row_names: list[str] = []
    for r in rows:
        name = f"{clean_backbone(r['backbone'])}\n{r['bad_type']}"
        if name not in row_names:
            row_names.append(name)
    values = np.zeros((len(row_names), len(components)))
    lookup = {
        (f"{clean_backbone(r['backbone'])}\n{r['bad_type']}", r["component"]): float(r["auc_high_score_is_clean"])
        for r in rows
    }
    for i, row_name in enumerate(row_names):
        for j, comp in enumerate(components):
            values[i, j] = lookup[(row_name, comp)]

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_title("Component Specificity by Bad-Augmentation Type")
    ax.set_xticks(np.arange(len(components)), [c.replace("_", "\n") for c in components], fontsize=8)
    ax.set_yticks(np.arange(len(row_names)), row_names, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AUC, higher score means cleaner")
    return save(fig, "figure3_component_specificity_heatmap.pdf")


def fig4_training_policy_non_promotion() -> Path:
    rows = read_csv(TABLES / "training_policy_summary.csv")
    labels = [r["branch"].replace("_", " ") for r in rows]
    metrics = [
        ("Delta BAcc", [float(r["delta_balanced_accuracy_vs_naive"]) for r in rows], "#4f81bd"),
        ("Delta Macro-F1", [float(r["delta_macro_f1_vs_naive"]) for r in rows], "#5f9e6e"),
        ("Delta ECE", [float(r["delta_ece_vs_naive"]) for r in rows], "#b94a48"),
    ]
    y = np.arange(len(labels))
    height = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, (name, vals, color) in enumerate(metrics):
        bars = ax.barh(y + (idx - 1) * height, vals, height=height, label=name, color=color)
        ax.bar_label(bars, labels=[f"{v:+.4f}" for v in vals], fontsize=8, padding=3)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axvline(0.01, color="#999999", linestyle=":", linewidth=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Delta versus NaiveAug")
    ax.set_title("Training Policy Deltas and Non-Promotion Boundary")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return save(fig, "figure4_training_policy_non_promotion.pdf")


def fig5_causal_chain() -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Supported and Unsupported Causal Chains", fontsize=15, weight="bold", pad=12)
    boxes = [
        (0.04, "Bad/clean\naugmentation\nis separable", "#e9f2fb", "#4f81bd"),
        (0.29, "Score direction\nmust be\naudited", "#e9f2fb", "#4f81bd"),
        (0.54, "Component\ndiagnostics\nrecover separation", "#e9f2fb", "#4f81bd"),
        (0.79, "Stable deployable\ntraining utility\nnot yet supported", "#fde9e7", "#b94a48"),
    ]
    for x, text, fc, ec in boxes:
        patch = FancyBboxPatch(
            (x, 0.38),
            0.17,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + 0.085, 0.52, text, ha="center", va="center", fontsize=10)
    for x0, x1, color, label in [(0.22, 0.29, "#333333", ""), (0.47, 0.54, "#333333", ""), (0.72, 0.79, "#b94a48", "not proven")]:
        ax.add_patch(FancyArrowPatch((x0, 0.52), (x1, 0.52), arrowstyle="->", mutation_scale=16, linewidth=1.4, color=color))
        if label:
            ax.text((x0 + x1) / 2, 0.61, label, ha="center", va="bottom", fontsize=9, color=color)
    return save(fig, "figure5_causal_chain.pdf")


def main() -> None:
    paths = [
        fig1_certificate_overview(),
        fig2_diagnostic_auc(),
        fig3_component_specificity_heatmap(),
        fig4_training_policy_non_promotion(),
        fig5_causal_chain(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
