# steegformer_sascert_core

## Intent

Validate whether the locked SAS-Cert method transfers from the CBraMod anchor to the stronger ST-EEGFormer-small backbone on PhysioNetMI.

Scientific question:

> Can SAS-Cert distinguish beneficial subject-style augmentation from harmful content/physiology/artifact distortion during few-shot cross-subject EEG-FM adaptation?

## Locked Scope

- Dataset: PhysioNetMI / EEGMMI
- Task: left vs right motor imagery
- Runs: R04 / R08 / R12
- Backbone: ST-EEGFormer-small
- Method: `SASCert_SoftAR_LS010`
- Seeds: 20, 21, 22, 23, 24
- Support: 5-shot per class for each target subject
- Target test: held-out target-subject trials only

## Groups

- `NaiveAug_LS010`
- `ArtifactReject_LS010`
- `SoftWeight_noReject_LS010`
- `SASCert_SoftAR_LS010`

## Fixed Method Parameters

- `artifact_reject_percentile = 10`
- `w_min = 0.2`
- `label_smoothing = 0.10`
- `score_variant = artifact_gate_content_rank`

## Go Criteria

Against `NaiveAug_LS010`:

- `delta_macro_f1 >= +0.005`
- `delta_balanced_acc >= 0.0` or no meaningful drop
- `delta_ece <= +0.01`
- `delta_nll <= +0.01`
- `delta_brier <= +0.01`
- `subject_win_rate_macro_f1 >= 0.65`
- `seed_win_rate_macro_f1 >= 0.65`

## Prohibited

- No MIRepNet
- No EEGPT
- No paper reproduction
- No hard Top50 mainline
- No target test leakage into ranking, thresholds, score normalization, checkpoint, or seed selection
- No hyperparameter tuning of locked SAS-Cert parameters

## Implementation Plan

1. Promote reusable loader / ST wrapper / metrics into `sas_core`.
2. Build the minimal ST-SAS-Cert runner inside this workbench.
3. Run a smoke test on one target subject and one seed.
4. Run the full target-subject x seed matrix.
5. If failure occurs, use `docs/FAILURE_REVIEW_PROTOCOL.md` for exactly one focused review cycle.

## Commands

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_steegformer_sascert_core/runner.py \
  --smoke --targets 90 --seeds 20 --device cuda

PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_steegformer_sascert_core/runner.py \
  --targets 90 91 92 --seeds 20 21 --device cuda \
  --source-epochs 30 --finetune-epochs 80 --batch-size 64

PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_steegformer_sascert_core/runner.py \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64
```

## Results

### Mini Matrix

| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Delta Brier |
|---|---:|---:|---:|---:|---:|
| SASCert vs Naive | +0.0255 | +0.0247 | -0.0115 | -0.0187 | -0.0124 |

Win rates:

- Subject win rate on Macro-F1: `0.3333`
- Seed win rate on Macro-F1: `1.0000`

Interpretation:

- The average signal is positive.
- Calibration also improves on average.
- Subject win rate is too low to claim support.
- Full target-subject x seed run is required.

### Full Matrix With Original Pretrained Frozen ST Features

Scope:

- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Rows: `400`
- Feature extractor: official ST-EEGFormer-small pretrained checkpoint frozen.

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.5039 | 0.4921 | 0.2649 | 0.8854 | 0.6172 |
| ArtifactReject_LS010 | 0.5067 | 0.4948 | 0.2676 | 0.8840 | 0.6168 |
| SoftWeight_noReject_LS010 | 0.5106 | 0.4982 | 0.2642 | 0.8876 | 0.6187 |
| SASCert_SoftAR_LS010 | 0.5161 | 0.5037 | 0.2619 | 0.8842 | 0.6170 |

SASCert vs Naive:

| Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Delta Brier | Subject Win Rate | Seed Win Rate |
|---:|---:|---:|---:|---:|---:|---:|
| +0.0121 | +0.0116 | -0.0030 | -0.0011 | -0.0002 | 0.25 | 0.40 |

Interpretation:

- The mean effect is positive and calibration does not worsen.
- The full Go criteria failed because subject and seed win rates are too low.
- More importantly, all four groups are near chance. This means the first broken link is likely the feature substrate, not only the SAS-Cert gate.

Primary outputs:

- `outputs/compact_steegformer_sascert_result.json`
- `outputs/steegformer_physionetmi_sascert_metrics.csv`
- `outputs/steegformer_physionetmi_paired_comparison.csv`
- `FAILURE_REVIEW.md`
- `failure_review_summary.json`

## Failure Review

Decision: `revise_training`

First broken link:

```text
pretrained frozen ST features are not sufficiently class-separable for this few-shot protocol
```

Next focused rerun:

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_steegformer_sascert_core/runner.py \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407
```

The runner now separates feature caches by tag, so source-tuned features will not reuse the original pretrained feature cache.

## Source-Tuned Feature Rerun

Feature extractor:

- `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Loaded into ST-EEGFormer-small as frozen source/validation-trained feature extractor.
- Load audit: `loaded_keys=106`, `missing_after_load=[]`, `unexpected_after_load=[]`.
- Feature tag: `st_source_tuned_seed3407`
- Output tag: `st_source_tuned_full`

Full scope:

- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Rows: `400`

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7088 | 0.7045 | 0.2079 | 0.6853 | 0.4159 |
| ArtifactReject_LS010 | 0.7107 | 0.7064 | 0.2112 | 0.6854 | 0.4165 |
| SoftWeight_noReject_LS010 | 0.7153 | 0.7109 | 0.2082 | 0.6832 | 0.4103 |
| SASCert_SoftAR_LS010 | 0.7149 | 0.7108 | 0.2056 | 0.6810 | 0.4097 |

SASCert vs Naive:

| Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Delta Brier | Subject Win Rate | Seed Win Rate |
|---:|---:|---:|---:|---:|---:|---:|
| +0.0061 | +0.0063 | -0.0023 | -0.0044 | -0.0062 | 0.25 | 0.00 |

SASCert vs components:

- Beats `ArtifactReject_LS010` on mean Macro-F1 by `+0.0044`.
- Does not beat `SoftWeight_noReject_LS010` on mean Macro-F1: `-0.0001`.

Interpretation:

- The source-tuned ST checkpoint repaired the feature-substrate problem: mean BAcc moved from near chance (`~0.516`) to useful few-shot adaptation (`~0.715`).
- Locked `SASCert_SoftAR_LS010` still does not meet reliability criteria because subject and seed win rates remain too low.
- For ST-EEGFormer-small, the artifact reject gate may be over-pruning or adding instability; `SoftWeight_noReject_LS010` is now the immediate branch to compare, while SoftAR remains useful for calibration/NLL.

Source-tuned outputs:

- `outputs/compact_steegformer_sascert_result_st_source_tuned_full.json`
- `outputs/steegformer_physionetmi_sascert_metrics_st_source_tuned_full.csv`
- `outputs/steegformer_physionetmi_paired_comparison_st_source_tuned_full.csv`
- `outputs/source_tuned_full.log`

## Artifact Gate Diagnostic

Diagnostic script:

- `diagnose_artifact_gate.py`

Inputs:

- Score rows: `outputs/score_rows/st_source_tuned_full/`
- Paired metrics: `outputs/steegformer_physionetmi_paired_comparison_st_source_tuned_full.csv`

Threshold sweep:

| Gate Percentile | Reject Rate | Clean Reject Rate | BadArtifact Reject Rate | Rejected Clean Fraction | Decision |
|---:|---:|---:|---:|---:|---|
| 90 | 0.10 | 0.00 | 0.50 | 0.00 | `artifact_gate_precise_but_conservative` |
| 80 | 0.20 | 0.00 | 1.00 | 0.00 | `artifact_gate_calibration_tradeoff` |
| 70 | 0.30 | 0.25 | 1.00 | 0.33 | `artifact_gate_overprunes_clean_or_useful_candidates` |

Interpretation:

- The p90 artifact gate is not falsely rejecting clean samples, but it only captures half of the synthetic BadArtifact candidates.
- The p80 threshold captures all BadArtifact candidates without rejecting clean samples in score-space diagnostics.
- The p70 threshold starts rejecting clean candidates and should not be promoted.

Mini p80 training check:

- Command used the same source-tuned ST checkpoint and the same mini scope `targets=90,91,92`, `seeds=20,21`.
- `SASCert_SoftAR_LS010` with p80 vs Naive:
  - Delta BAcc `-0.0080`
  - Delta Macro-F1 `-0.0081`
  - Delta ECE `-0.0141`
  - Delta NLL `+0.0024`
  - Subject win rate Macro-F1 `0.00`
  - Seed win rate Macro-F1 `0.00`

Conclusion:

- p80 looks cleaner as a detector but hurts mini classification, so do not spend a full 20-subject run on it yet.
- The current ST branch should favor `SoftWeight_noReject_LS010` for classification, while artifact gating remains a calibration-oriented option.

Artifact diagnostic outputs:

- `outputs/artifact_gate_diagnostics/ARTIFACT_GATE_DIAGNOSTIC_st_source_tuned_full_p90.md`
- `outputs/artifact_gate_diagnostics/ARTIFACT_GATE_DIAGNOSTIC_st_source_tuned_full_p80.md`
- `outputs/artifact_gate_diagnostics/ARTIFACT_GATE_DIAGNOSTIC_st_source_tuned_full_p70.md`
- `outputs/compact_steegformer_sascert_result_st_source_tuned_p80_mini.json`

## Decision

Current decision: `revise_method_after_source_tuned_rerun`

Updated method-local decision:

```text
prefer_softweight_no_reject_for_st_classification; keep_artifact_gate_as_calibration_diagnostic
```

## Component-Gated Reliability Mini

Purpose:

- Test whether score-validated `component_gated_v1` and
  `score_artifact_gate_physio` become useful training weights for
  source-tuned `ST-EEGFormer-small`.
- Use the same mini scope as the cross-backbone direction audit:
  targets `90,91,92`, seeds `20,21`.

Command:

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_steegformer_sascert_core/runner.py \
  --targets 90 91 92 --seeds 20 21 \
  --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag st_source_tuned_component_gated_mini \
  --experiment component_gated
```

Result:

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7869 | 0.7862 | 0.1413 | 0.5065 | 0.3135 |
| `SoftWeight_noReject_LS010` | 0.7902 | 0.7898 | 0.1471 | 0.4908 | 0.2959 |
| `SASCert_SoftAR_LS010` | 0.7860 | 0.7852 | 0.1459 | 0.4966 | 0.3012 |
| `ArtifactGatePhysio_LS010` | 0.7823 | 0.7817 | 0.1385 | 0.5048 | 0.3151 |
| `ComponentGatedV1_LS010` | 0.7823 | 0.7817 | 0.1473 | 0.5093 | 0.3178 |

Decision:

```text
do_not_expand_component_gated_or_artifact_gate_physio_on_st
```

Interpretation:

- `component_gated_v1` and `artifact_gate_physio` are strong score-only
  diagnostics, but they did not improve mini training on ST.
- The current best ST training branch remains `SoftWeight_noReject_LS010`.
- Next ST work should confirm/report the no-reject branch instead of adding
  another gate variant.

Outputs:

- `outputs/COMPONENT_GATED_ST_RELIABILITY_MINI.md`
- `outputs/compact_steegformer_sascert_result_st_source_tuned_component_gated_mini.json`
- `outputs/steegformer_physionetmi_sascert_metrics_st_source_tuned_component_gated_mini.csv`
- `outputs/steegformer_physionetmi_paired_comparison_st_source_tuned_component_gated_mini.csv`

## SoftWeight No-Reject Locked Confirm

Purpose:

- Package the strongest current ST training branch into a locked confirmation
  report instead of relying on scattered JSON files.
- Decide whether `SoftWeight_noReject_LS010` can be promoted as the main ST
  method.

Outputs:

- `outputs/locked_confirm/ST_SOFTWEIGHT_NO_REJECT_LOCKED_CONFIRM.md`
- `outputs/locked_confirm/compact_softweight_locked_confirm.json`
- `outputs/locked_confirm/SOFTWEIGHT_FAILURE_REVIEW.md`
- `outputs/locked_confirm/softweight_failure_review_summary.json`

Key result:

| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Positive-Mean Subject Rate | Majority-Seed Subject Win Rate | Seed Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SoftWeight - Naive` | +0.0065 | +0.0064 | +0.0003 | -0.0022 | 0.60 | 0.15 | 0.00 |
| `SoftWeight - SoftAR` | +0.0003 | +0.0001 | +0.0026 | +0.0022 | 0.50 | 0.05 | 0.00 |

Decision:

```text
do_not_promote_softweight_no_reject
```

Failure review decision:

```text
revise_training
```

First broken link:

```text
clean-vs-bad certificate quality
  -> subject/seed-stable training utility
```

Next allowed diagnostic:

```text
support_candidate_utility_alignment_audit
```

Stop rule:

- Only one existing-output utility-alignment audit is allowed before deciding
  whether to revise the weighting rule with a locked support-only hypothesis or
  park ST weighting variants.

## Support/Candidate Utility Alignment Audit

Purpose:

- Use existing `st_source_tuned_full` score rows and metrics to test whether
  candidate-level summaries explain `SoftWeight_noReject_LS010` benefit.
- Do not train or tune any new method.

Outputs:

- `outputs/utility_alignment_audit/SUPPORT_CANDIDATE_UTILITY_ALIGNMENT_AUDIT.md`
- `outputs/utility_alignment_audit/compact_utility_alignment_audit.json`
- `outputs/utility_alignment_audit/fold_utility_alignment_features.csv`
- `outputs/utility_alignment_audit/utility_alignment_correlations.csv`

Result:

| Strongest candidate-only feature | Spearman with SoftWeight benefit | Threshold |
|---|---:|---:|
| `clean_artifact_risk_raw_mean` | 0.1168 | 0.35 |

Decision:

```text
park_st_weighting_variants
```

Interpretation:

- Candidate-only score summaries do not explain which target folds benefit
  from SoftWeight.
- The retrospective outcome variables correlate with each other, but those are
  not legal predictors for a future support-only rule.
- Under the stop rule, ST weighting variants should now be parked or reframed
  as diagnostic observations rather than expanded.
