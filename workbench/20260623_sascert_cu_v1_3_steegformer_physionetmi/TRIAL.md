# sascert_cu_v1_3_steegformer_physionetmi

## Intent

Repair v1.2 by decoupling the multidimensional diagnostic certificate from the
training utility policy. v1.3 keeps artifact / physio / style / prediction
scores for audit reports only, while the main training weight uses content
utility from embedding and prototype consistency.

## Protocol

- Backbone: ST-EEGFormer-small.
- Dataset: PhysioNetMI / EEGMMI left-vs-right motor imagery.
- Runs: R04/R08/R12.
- Source-tuned checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`.
- Targets: 90-109.
- Seeds: 20-24.
- Groups:
  - `NaiveAug_LS010`
  - `SoftWeight_noReject_LS010`
  - `SAS-Cert-CU-LS-v1.3`
- Weight:
  - `E_content = ranknorm(E_embed) + ranknorm(E_proto)`
  - `w = 0.75 + 0.5 * ranknorm(E_content)`
- Loss:
  - `L_aug = sum(w_i * CE_i) / (sum(w_i) + eps)`
  - `L = CE_real + L_aug`
  - label smoothing `0.10`
- Leakage boundary:
  - ranknorm and prototypes use only train/support candidates.
  - target test is final evaluation only.

## Commands

```bash
python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --experiment v1_3 \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_3 \
  --output-dir workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs
```

## Required Outputs

- `outputs/SASCERT_CU_V1_3_REPORT.md`
- `outputs/compact_sascert_v1_3_result.json`
- `outputs/metrics_v1_3.csv`
- `outputs/paired_comparison_v1_3.csv`
- `outputs/component_utility_audit.csv`
- `outputs/component_utility_summary.json`
- `outputs/diagnostic_scores_v1_3.csv`
- `outputs/leakage_audit_v1_3.json`

## Results

Completed full run over targets 90-109 and seeds 20-24.

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7567 | 0.7524 | 0.8212 | 0.1857 | 0.6221 | 0.3594 | 0.0000 | 1.0000 |
| `SoftWeight_noReject_LS010` | 0.7568 | 0.7527 | 0.8255 | 0.1826 | 0.6150 | 0.3559 | 0.0000 | 0.6000 |
| `SAS-Cert-CU-LS-v1.3` | 0.7588 | 0.7543 | 0.8235 | 0.1861 | 0.6206 | 0.3587 | 0.0000 | 1.0000 |

v1.3 vs `NaiveAug_LS010`:

- Delta BAcc: `+0.002078`
- Delta Macro-F1: `+0.001935`
- Delta ECE: `+0.000444`
- Delta NLL: `-0.001490`
- Delta Brier: `-0.000763`

v1.3 vs `SoftWeight_noReject_LS010`:

- Delta BAcc: `+0.001959`
- Delta Macro-F1: `+0.001634`
- Delta ECE: `+0.003503`
- Delta NLL: `+0.005643`
- Delta Brier: `+0.002736`

Component Utility Audit:

- Training utility candidates: `E_embed`, `E_proto`, `E_content`.
- Diagnostic-only or unstable for training utility:
  `E_pred`, `artifact_score`, `artifact_safe`, `E_physio`, `E_style`,
  `D_band`, `D_cov`, `D_style`.
- Strongest content signal:
  - `E_proto`: Spearman CE `-0.8877`, correctness `+0.4623`
  - `E_content`: Spearman CE `-0.5293`, correctness `+0.2972`

Leakage audit: `passed`.

## Decision

- `enter_cbramod_recheck`
- Reason: v1.3 is positive vs both NaiveAug and SoftWeight on BAcc/Macro-F1,
  with small calibration trade-off vs SoftWeight.
