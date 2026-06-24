# sascert_softsafe_v1_2_steegformer_physionetmi

## Intent

Repair v1.1 by replacing hard top-10 artifact rejection with SoftSafe artifact penalty and normalized augmentation loss.

## Runner Reuse

This trial intentionally reuses and iterates the existing runner:

`workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py`

No duplicate runner was created.

## Outputs

## Commands

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --experiment v1_2 \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --device cuda --source-epochs 30 --finetune-epochs 80 \
  --batch-size 64 --feature-batch-size 64 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_2 \
  --output-dir workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs
```

## Results

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7101 | 0.7052 | 0.7810 | 0.2156 | 0.7338 | 0.4264 | 0.0000 | 1.0000 |
| `SoftWeight_noReject_LS010` | 0.7100 | 0.7049 | 0.7812 | 0.2156 | 0.7327 | 0.4241 | 0.0000 | 0.6000 |
| `SAS-Cert-SoftSafe-LS-v1.2` | 0.7070 | 0.7019 | 0.7829 | 0.2135 | 0.7340 | 0.4251 | 0.0965 | 0.6868 |

Primary deltas:

- v1.2 vs Naive:
  - BAcc `-0.0030`
  - Macro-F1 `-0.0032`
  - ECE `-0.0021`
  - NLL `+0.0002`
  - Brier `-0.0014`
- v1.2 vs SoftWeight no-reject:
  - BAcc `-0.0030`
  - Macro-F1 `-0.0030`
  - ECE `-0.0021`
  - NLL `+0.0013`
  - Brier `+0.0010`

Audits:

- v1.1 gate harm:
  - rejected content mean `0.5404`
  - kept content mean `0.4955`
  - rejected physio mean `0.5451`
  - kept physio mean `0.4950`
  - conclusion: v1.1 hard gate did reject relatively high-content/high-physio
    candidates.
- loss mass:
  - v1.1 SoftAR effective augmentation scale `0.5351`
  - v1.2 SoftSafe effective augmentation scale `0.6868`
  - conclusion: v1.2 repaired the low weight-mass issue, but that did not
    recover ST classification utility.
- leakage audit: passed.

## Decision

`ARTIFACT_PHYSIO_STYLE_NOT_HELPING_ST`

v1.2 improves calibration slightly but does not beat Naive or SoftWeight on
BAcc/Macro-F1. The next recommended branch is content-only soft weighting, not
CBraMod replication of the current SoftSafe policy.

## Outputs

- `outputs/SASCERT_SOFTSAFE_V1_2_REPORT.md`
- `outputs/compact_sascert_v1_2_result.json`
- `outputs/metrics_v1_2.csv`
- `outputs/paired_comparison_v1_2.csv`
- `outputs/gate_harm_audit.csv`
- `outputs/loss_mass_audit.csv`
- `outputs/certificate_distribution_v1_2.csv`
- `outputs/leakage_audit_v1_2.json`
