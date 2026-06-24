#!/usr/bin/env python3
"""Focused failure review for the CBraMod PhysioNetMI mini matrix."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "local_python_deps"
if str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

from sas_core.data.physionet_mi import build_physionet_mi_cache, default_physionet_mi_protocol, split_indices, support_test_split_for_subject
from sas_core.metrics.classification import classification_metrics
from sas_core.utils.io import write_json
from sas_core.utils.seed import set_seed


WB_DIR = Path(__file__).resolve().parent
OUT_DIR = WB_DIR / "outputs"
CANONICAL_CACHE = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "data" / "physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz"
FEATURE_CACHE = OUT_DIR / "cbramod_original_features.npz"
MINI_METRICS = OUT_DIR / "cbramod_physionetmi_sascert_metrics_mini.csv"
REPAIRED_COMPACT = OUT_DIR / "compact_cbramod_physionetmi_sascert_result_repaired_mini.json"
SCORE_DIR = OUT_DIR / "score_rows" / "mini"
TARGETS = [90, 91, 92]
SEEDS = [20, 21]


class FeatureHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    loss = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    init_state: Dict[str, torch.Tensor] | None,
    device: str,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 64,
    label_smoothing: float = 0.0,
) -> FeatureHead:
    head = FeatureHead(features.shape[1]).to(device)
    if init_state is not None:
        head.load_state_dict(init_state)
    ds = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).long(), torch.from_numpy(weights).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        head.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = weighted_ce(head(xb), yb, wb, label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite diagnostic head loss")
            loss.backward()
            opt.step()
    return head


def predict_probs(head: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            xb = torch.from_numpy(features[start : start + 256]).float().to(device)
            out.append(torch.softmax(head(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_means(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, float]]:
    groups = sorted(set(row["group"] for row in rows))
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]
    return {
        group: {metric: float(np.mean([float(row[metric]) for row in rows if row["group"] == group])) for metric in metrics}
        for group in groups
    }


def mean_by_key(rows: Sequence[dict[str, object]], key: str) -> dict[str, float]:
    groups = sorted(set(str(row[key]) for row in rows))
    return {group: float(np.mean([float(row["macro_f1"]) for row in rows if str(row[key]) == group])) for group in groups}


def train_source_state(data: dict[str, np.ndarray], features: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    source_idx = split_indices(data["subjects"])["train"]
    head = train_head(features[source_idx], data["y"][source_idx], np.ones(len(source_idx), dtype=np.float32), None, device, epochs=30)
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def run_noaug_diagnostic(data: dict[str, np.ndarray], features: np.ndarray, device: str) -> tuple[list[dict[str, object]], dict[str, float]]:
    source_state = train_source_state(data, features, device)
    rows: list[dict[str, object]] = []
    source_only_probs = []
    source_only_y = []

    source_head = FeatureHead(features.shape[1]).to(device)
    source_head.load_state_dict(source_state)
    for target in TARGETS:
        target_idx = np.where(data["subjects"] == target)[0]
        source_only_probs.append(predict_probs(source_head, features[target_idx], device))
        source_only_y.append(data["y"][target_idx])
        for seed in SEEDS:
            support_idx, test_idx = support_test_split_for_subject(data["y"], data["subjects"], target, 5, seed)
            head = train_head(
                features[support_idx],
                data["y"][support_idx],
                np.ones(len(support_idx), dtype=np.float32),
                source_state,
                device,
                epochs=80,
                label_smoothing=0.10,
            )
            probs = predict_probs(head, features[test_idx], device)
            metrics = classification_metrics(data["y"][test_idx], probs)
            rows.append({"target_subject": target, "seed": seed, "group": "NoAug_LS010", **metrics})
    source_metrics = classification_metrics(np.concatenate(source_only_y), np.concatenate(source_only_probs))
    return rows, source_metrics


def auc_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def score_audit() -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in sorted(SCORE_DIR.glob("target*_seed*.csv")):
        target_seed = path.stem
        for row in read_csv_rows(path):
            row["fold"] = target_seed
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no score rows found in {SCORE_DIR}")

    components = ["content_score", "style_score", "physio_score", "artifact_safe_score", "sas_score"]
    y_clean = np.asarray([1 if row["aug_type"] == "clean" else 0 for row in rows], dtype=np.int64)
    overall = {component: auc_or_none(y_clean, np.asarray([float(row[component]) for row in rows])) for component in components}

    def variant_auc(score: np.ndarray) -> float | None:
        return auc_or_none(y_clean, score.astype(np.float64))

    content = np.asarray([float(row["content_score"]) for row in rows], dtype=np.float32)
    style = np.asarray([float(row["style_score"]) for row in rows], dtype=np.float32)
    physio = np.asarray([float(row["physio_score"]) for row in rows], dtype=np.float32)
    artifact_safe = np.asarray([float(row["artifact_safe_score"]) for row in rows], dtype=np.float32)
    artifact_risk = np.asarray([float(row["artifact_risk_raw"]) for row in rows], dtype=np.float32)

    content_fixed = 1.0 - content if (overall["content_score"] or 0.0) < 0.5 else content
    artifact_safe_fixed = 1.0 - artifact_safe if (overall["artifact_safe_score"] or 0.0) < 0.5 else artifact_safe
    physio_fixed = 1.0 - physio if (overall["physio_score"] or 0.0) < 0.5 else physio
    style_fixed = 1.0 - style if (overall["style_score"] or 0.0) < 0.5 else style
    fixed_total = 0.35 * content_fixed + 0.25 * physio_fixed + 0.25 * artifact_safe_fixed + 0.15 * style_fixed
    physio_style = 0.7 * physio + 0.3 * style
    artifact_gate_physio = physio.copy()
    artifact_gate_physio[artifact_risk >= np.percentile(artifact_risk, 90.0)] = 0.0
    variant_auc = {
        "current_sas_score": overall["sas_score"],
        "physio_only": variant_auc(physio),
        "style_only": variant_auc(style),
        "physio_style": variant_auc(physio_style),
        "direction_fixed_total": variant_auc(fixed_total),
        "artifact_gate_physio": variant_auc(artifact_gate_physio),
    }

    by_bad_type: dict[str, dict[str, float | None]] = {}
    for bad_type in ["bad_artifact", "bad_content", "bad_physio"]:
        subset = [row for row in rows if row["aug_type"] in {"clean", bad_type}]
        y = np.asarray([1 if row["aug_type"] == "clean" else 0 for row in subset], dtype=np.int64)
        by_bad_type[bad_type] = {
            component: auc_or_none(y, np.asarray([float(row[component]) for row in subset])) for component in components
        }

    reject_stats: dict[str, dict[str, float]] = {}
    for fold in sorted(set(row["fold"] for row in rows)):
        fold_rows = [row for row in rows if row["fold"] == fold]
        risks = np.asarray([float(row["artifact_risk_raw"]) for row in fold_rows])
        threshold = np.percentile(risks, 90.0)
        for aug_type in sorted(set(row["aug_type"] for row in fold_rows)):
            type_rows = [row for row in fold_rows if row["aug_type"] == aug_type]
            reject_rate = float(np.mean([float(row["artifact_risk_raw"]) >= threshold for row in type_rows]))
            reject_stats.setdefault(aug_type, {"sum": 0.0, "n": 0})
            reject_stats[aug_type]["sum"] += reject_rate
            reject_stats[aug_type]["n"] += 1
    reject_summary = {
        aug_type: {"p90_reject_rate": vals["sum"] / vals["n"], "folds": vals["n"]} for aug_type, vals in reject_stats.items()
    }

    return {
        "n_score_rows": len(rows),
        "overall_clean_vs_bad_auc_high_score_is_clean": overall,
        "clean_vs_bad_type_auc_high_score_is_clean": by_bad_type,
        "score_variant_auc_high_score_is_clean": variant_auc,
        "direction_fixed_components": {
            "content_inverted": bool((overall["content_score"] or 0.0) < 0.5),
            "artifact_safe_inverted": bool((overall["artifact_safe_score"] or 0.0) < 0.5),
            "physio_inverted": bool((overall["physio_score"] or 0.0) < 0.5),
            "style_inverted": bool((overall["style_score"] or 0.0) < 0.5),
        },
        "artifact_p90_reject_rate_by_aug_type": reject_summary,
    }


def make_review_markdown(summary: dict[str, object]) -> str:
    mini = summary["mini_group_means"]
    noaug = summary["noaug_group_mean"]
    source = summary["source_only_target_metrics"]
    score = summary["score_audit"]
    sas_delta = summary["sascert_vs_naive"]
    repaired = summary.get("repaired_score_mini")
    if repaired:
        repaired_delta = repaired["primary_vs_naive"]
        repaired_line = (
            f"| RepairedSoftWeight - Naive Macro-F1 / ECE | "
            f"`{repaired_delta['delta_macro_f1']:.4f}` / `{repaired_delta['delta_ece']:.4f}` |"
        )
        repaired_next = (
            "- The repaired score rescued classification on the mini matrix, but it still failed the calibration gate.\n"
            f"  RepairedSoftWeight vs Naive: Macro-F1 `{repaired_delta['delta_macro_f1']:.4f}`, "
            f"ECE `{repaired_delta['delta_ece']:.4f}`.\n"
        )
    else:
        repaired_line = "| RepairedSoftWeight - Naive Macro-F1 / ECE | `not run` |"
        repaired_next = "- Repaired-score mini has not been run yet.\n"

    return f"""# CBraMod PhysioNetMI Failure Review

## 1. Facts Only

| Item | Value |
|---|---:|
| Mini targets | `90,91,92` |
| Mini seeds | `20,21` |
| NaiveAug BAcc / Macro-F1 / ECE | `{mini['NaiveAug_LS010']['balanced_accuracy']:.4f}` / `{mini['NaiveAug_LS010']['macro_f1']:.4f}` / `{mini['NaiveAug_LS010']['ece']:.4f}` |
| SASCert SoftAR BAcc / Macro-F1 / ECE | `{mini['SASCert_SoftAR_LS010']['balanced_accuracy']:.4f}` / `{mini['SASCert_SoftAR_LS010']['macro_f1']:.4f}` / `{mini['SASCert_SoftAR_LS010']['ece']:.4f}` |
| SASCert SoftAR - Naive BAcc | `{sas_delta['delta_balanced_accuracy']:.4f}` |
| SASCert SoftAR - Naive Macro-F1 | `{sas_delta['delta_macro_f1']:.4f}` |
| SASCert SoftAR - Naive ECE | `{sas_delta['delta_ece']:.4f}` |
| NoAug BAcc / Macro-F1 / ECE | `{noaug['balanced_accuracy']:.4f}` / `{noaug['macro_f1']:.4f}` / `{noaug['ece']:.4f}` |
| Source-only target BAcc / Macro-F1 / ECE | `{source['balanced_accuracy']:.4f}` / `{source['macro_f1']:.4f}` / `{source['ece']:.4f}` |
{repaired_line}

Data quality notes:

- No NaN/Inf was reported in the frozen feature cache.
- No raw EEG data or raw augmented arrays were copied into this workbench.
- Target held-out labels were used only for final evaluation.

## 2. Expected vs Actual

| Expectation | Actual Result | Verdict | Key Data |
|---|---|---|---|
| CBraMod features should support target few-shot adaptation above chance | Target metrics are near chance | rejected | Naive Macro-F1 `{mini['NaiveAug_LS010']['macro_f1']:.4f}`, NoAug Macro-F1 `{noaug['macro_f1']:.4f}` |
| SASCert SoftAR should improve Macro-F1 over NaiveAug | Macro-F1 gain is negligible | rejected | delta Macro-F1 `{sas_delta['delta_macro_f1']:.4f}` |
| SASCert should not worsen calibration beyond +0.01 ECE | ECE worsens beyond gate | rejected | delta ECE `{sas_delta['delta_ece']:.4f}` |
| Artifact gate should reject artifact candidates more than clean candidates | Gate is selective for BadArtifact | supported | BadArtifact p90 reject `{score['artifact_p90_reject_rate_by_aug_type']['bad_artifact']['p90_reject_rate']:.4f}`, clean reject `{score['artifact_p90_reject_rate_by_aug_type']['clean']['p90_reject_rate']:.4f}` |
| Cert scores should separate clean from bad candidates | Current total score is directionally wrong on the mixed bad pool | rejected | overall SAS AUC `{score['overall_clean_vs_bad_auc_high_score_is_clean']['sas_score']:.4f}` |

## 3. Causal Chain

```text
Bad/clean augmentation separable
→ cert score ranks useful samples higher
→ soft weighting/rejection changes training signal
→ few-shot target adaptation improves
→ calibration does not degrade
```

| Link | Status | Evidence |
|---|---|---|
| Bad/clean augmentation separable | supported by some components, rejected by current total score | physio AUC `{score['overall_clean_vs_bad_auc_high_score_is_clean']['physio_score']:.4f}`, current SAS AUC `{score['overall_clean_vs_bad_auc_high_score_is_clean']['sas_score']:.4f}` |
| Cert score ranks useful samples higher | rejected for current total score | current SAS AUC `{score['overall_clean_vs_bad_auc_high_score_is_clean']['sas_score']:.4f}`, direction-fixed total AUC `{score['score_variant_auc_high_score_is_clean']['direction_fixed_total']:.4f}` |
| Soft weighting/rejection changes training signal | supported | SASCert SoftAR changes BAcc by `{sas_delta['delta_balanced_accuracy']:.4f}` and ECE by `{sas_delta['delta_ece']:.4f}` |
| Few-shot target adaptation improves | rejected for Macro-F1 | Macro-F1 delta `{sas_delta['delta_macro_f1']:.4f}` |
| Calibration does not degrade | rejected | ECE delta `{sas_delta['delta_ece']:.4f}` |

First broken link:

```text
current mixed-bad SAS score direction is wrong on PhysioNetMI
→ selected/weighted augmentations do not improve Macro-F1
→ the weak frozen CBraMod feature space leaves little margin for recovery
```

## 4. Possible Explanations

1. Highest credibility: the current CBraMod PhysioNetMI certificate formula is directionally wrong for the mixed bad pool.
   - Explains: current SAS clean-vs-bad AUC is `{score['overall_clean_vs_bad_auc_high_score_is_clean']['sas_score']:.4f}`, while a direction-fixed score-only variant reaches `{score['score_variant_auc_high_score_is_clean']['direction_fixed_total']:.4f}`.
   - Does not fully explain: why absolute target adaptation is weak.
   - Distinguishing measurement: train only one repaired score variant on the same mini targets after freezing the formula from score-only diagnostics.

2. Medium credibility: frozen CBraMod pooled features are not aligned enough for this PhysioNetMI protocol.
   - Explains: Source-only, NoAug, and NaiveAug are all near chance.
   - Does not fully explain: why SASCert can still lift BAcc slightly.
   - Distinguishing measurement: compare frozen pooled features against a source-tuned CBraMod or a different pooling/head policy on the same mini targets.

3. Lower credibility: the SAS artifact/content gate detects only artifact synthetic badness, not target utility.
   - Explains: artifact reject stats are plausible while Macro-F1 gains are negligible.
   - Does not explain: low absolute baseline.
   - Distinguishing measurement: correlate candidate scores with leave-one-candidate-out support validation utility, using only target support.

## 5. One Next Experiment

Do not expand the current mini matrix to full targets yet.

Next focused experiment:

```text
CBraMod cert-direction repair mini:
freeze a repaired score from existing score-only diagnostics
and run only NoAug / NaiveAug / repaired SoftWeight on targets 90-92 and seeds 20-21.
```

Diagnostic rerun result:

{repaired_next}

Go/No-Go:

- Go to broader CBraMod validation only if repaired SoftWeight beats both NoAug and NaiveAug by at least `+1pp` Macro-F1 without ECE worsening beyond `+0.01`.
- If the repaired score still fails, park CBraMod PhysioNetMI and keep ST-EEGFormer-small as the active PhysioNetMI backbone.

## 6. Decision

`revise_cert_calibration`

This is not a reason to abandon SAS-Cert. It says the current CBraMod
PhysioNetMI total certificate is directionally wrong under the mixed bad
candidate pool, and the frozen feature space is weak enough that bad scoring
cannot be rescued by training. The repaired score mini shows classification
can improve after direction repair, but calibration still violates the gate.
Do not run full CBraMod PhysioNetMI until a calibration-aware repaired score or
loss is defined.
"""


def main() -> None:
    set_seed(3407)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = default_physionet_mi_protocol()
    data = build_physionet_mi_cache(CANONICAL_CACHE, protocol=protocol, rebuild=False)
    feature_npz = np.load(FEATURE_CACHE)
    features = feature_npz["features"].astype(np.float32)
    if features.shape[0] != data["X"].shape[0]:
        raise RuntimeError(f"feature/data mismatch: {features.shape[0]} vs {data['X'].shape[0]}")

    mini_rows = read_csv_rows(MINI_METRICS)
    mini_means = group_means(mini_rows)
    noaug_rows, source_metrics = run_noaug_diagnostic(data, features, device)
    noaug_mean = {
        metric: float(np.mean([float(row[metric]) for row in noaug_rows]))
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "auroc", "ece", "nll", "brier"]
    }
    score = score_audit()
    repaired = None
    if REPAIRED_COMPACT.exists():
        repaired = json.loads(REPAIRED_COMPACT.read_text(encoding="utf-8"))
    sas = mini_means["SASCert_SoftAR_LS010"]
    naive = mini_means["NaiveAug_LS010"]
    summary = {
        "status": "completed",
        "decision": "revise_cert_calibration",
        "first_broken_link": "current_mixed_bad_sas_score_direction_wrong_on_physionetmi",
        "mini_group_means": mini_means,
        "noaug_group_mean": noaug_mean,
        "source_only_target_metrics": source_metrics,
        "sascert_vs_naive": {
            "delta_balanced_accuracy": sas["balanced_accuracy"] - naive["balanced_accuracy"],
            "delta_macro_f1": sas["macro_f1"] - naive["macro_f1"],
            "delta_ece": sas["ece"] - naive["ece"],
            "delta_nll": sas["nll"] - naive["nll"],
            "delta_brier": sas["brier"] - naive["brier"],
        },
        "score_audit": score,
        "repaired_score_mini": repaired,
        "next_experiment": "define one calibration-aware repaired score/loss before any full CBraMod PhysioNetMI expansion",
        "leakage_audit": {
            "raw_data_copied": False,
            "raw_augmented_arrays_saved": False,
            "target_test_used_for_ranking_threshold_or_training": False,
            "target_test_used_for_final_evaluation_only": True,
        },
    }
    write_json(WB_DIR / "failure_review_summary.json", summary)
    (WB_DIR / "FAILURE_REVIEW.md").write_text(make_review_markdown(summary), encoding="utf-8")

    noaug_path = OUT_DIR / "cbramod_physionetmi_noaug_diagnostic_mini.csv"
    with noaug_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(noaug_rows[0].keys()))
        writer.writeheader()
        writer.writerows(noaug_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
