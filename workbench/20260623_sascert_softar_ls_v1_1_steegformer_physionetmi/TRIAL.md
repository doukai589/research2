# sascert_softar_ls_v1_1_steegformer_physionetmi

## Intent

Implement and validate `SAS-Cert-SoftAR-LS v1.1` on the locked
`ST-EEGFormer-small + PhysioNetMI` protocol.

Core question:

> Can a risk-controlled augmentation policy using safety gating,
> label-preservation evidence, physiology/style plausibility, utility weighting,
> and label smoothing improve reliable few-shot adaptation?

## Protocol

- Backbone: `ST-EEGFormer-small`
- Source-tuned checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Dataset: `PhysioNetMI / EEGMMI`
- Task: left-vs-right motor imagery
- Runs: `R04/R08/R12`
- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Support: 5-shot per class target support
- Candidate augmentations:
  - Gaussian noise
  - time shift
  - time crop
  - frequency mask
  - channel dropout
  - mild frequency mixup
- Fixed parameters:
  - artifact reject ratio: `10%`
  - `w_min = 0.2`
  - label smoothing: `0.10`
- Groups:
  - `NaiveAug_LS010`
  - `ArtifactReject_LS010`
  - `SoftWeight_noReject_LS010`
  - `SAS-Cert-SoftAR-LS-v1.1`
- Leakage rule:
  - target test is final-evaluation only.
  - artifact thresholds, rank normalization, prototypes, and style anchors are
    computed only from legal training/support/candidate data.

## Commands

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --smoke --targets 90 --seeds 20 --device cuda \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_1_smoke

PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --device cuda --source-epochs 30 --finetune-epochs 80 \
  --batch-size 64 --feature-batch-size 64 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_1_source_tuned_full
```

## Results

Full matrix:

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7156 | 0.7098 | 0.7825 | 0.2093 | 0.7208 | 0.4225 | 0.0000 | 1.0000 |
| `ArtifactReject_LS010` | 0.7074 | 0.7013 | 0.7823 | 0.2150 | 0.7304 | 0.4267 | 0.1000 | 1.0000 |
| `SoftWeight_noReject_LS010` | 0.7117 | 0.7050 | 0.7823 | 0.2123 | 0.7247 | 0.4238 | 0.0000 | 0.6000 |
| `SAS-Cert-SoftAR-LS-v1.1` | 0.7102 | 0.7047 | 0.7819 | 0.2105 | 0.7275 | 0.4246 | 0.1000 | 0.5351 |

Primary deltas:

- v1.1 vs Naive:
  - BAcc `-0.0054`
  - Macro-F1 `-0.0051`
  - ECE `+0.0012`
  - NLL `+0.0068`
  - Brier `+0.0021`
  - subject win rate Macro-F1 `0.25`
  - seed win rate Macro-F1 `0.00`
- v1.1 vs ArtifactReject:
  - BAcc `+0.0028`
  - Macro-F1 `+0.0033`
  - ECE `-0.0045`
  - NLL `-0.0028`
  - Brier `-0.0022`
- v1.1 vs SoftWeight no-reject:
  - BAcc `-0.0015`
  - Macro-F1 `-0.0003`
  - ECE `-0.0018`
  - NLL `+0.0029`
  - Brier `+0.0008`

Tool status:

- `pyRiemann`: used.
- `MNE-Features`: used.
- `Autoreject`: available for offline audit but not integrated into training.

Leakage audit:

- Passed; no target-test use in thresholds, ranknorm, prototypes, style anchor,
  best epoch, or best seed.

Main outputs:

- `outputs/SASCERT_SOFTAR_LS_V1_1_REPORT_v1_1_source_tuned_full.md`
- `outputs/compact_sascert_v1_1_result_v1_1_source_tuned_full.json`
- `outputs/steegformer_physionetmi_sascert_metrics_v1_1_source_tuned_full.csv`
- `outputs/steegformer_physionetmi_paired_comparison_v1_1_source_tuned_full.csv`
- `outputs/certificate_score_distribution_v1_1_source_tuned_full.csv`
- `outputs/rejected_samples_summary_v1_1_source_tuned_full.csv`
- `outputs/failure_cases_summary_v1_1_source_tuned_full.csv`
- `outputs/leakage_audit_v1_1_source_tuned_full.json`

## Decision

- `do_not_promote_v1_1_on_st`

Interpretation:

- v1.1 does not beat `NaiveAug_LS010`.
- v1.1 beats standalone `ArtifactReject_LS010`, so soft utility weighting
  partially repairs the gate-only policy.
- v1.1 does not beat `SoftWeight_noReject_LS010`, meaning the artifact gate is
  still the unstable component on ST.
- Do not migrate this v1.1 policy to CBraMod yet as a success claim. Either
  revise the gate/weight interaction under a new locked hypothesis or park ST
  v1.1 as a negative validation.
