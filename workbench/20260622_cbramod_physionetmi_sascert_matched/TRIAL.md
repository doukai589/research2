# cbramod_physionetmi_sascert_matched

## Intent

Run the CBraMod anchor on the same PhysioNetMI few-shot SAS-Cert protocol used
for ST-EEGFormer-small.

Scientific question:

> Does augmented-sample certification behave consistently across two EEG
> foundation-model backbones when the dataset and few-shot target-support
> protocol are held fixed?

## Locked Scope

- Dataset: `PhysioNetMI`
- Task: left vs right motor imagery
- Runs: `R04/R08/R12`
- Backbone: `CBraMod`
- Backbone training: frozen encoder only
- Source subjects: `1-70`
- Validation subjects: `71-89`
- Final target subjects: `90-109`
- Support: 5-shot per class
- Seeds: `20,21,22,23,24`

## Reference ST Outcome

The matched ST-EEGFormer branch reached a diagnostic pause point:

- Source-tuned ST features were useful.
- `SoftWeight_noReject_LS010` and `SASCert_SoftAR_LS010` both improved mean
  metrics over `NaiveAug_LS010`.
- Neither method was reliable enough across subjects/seeds.
- Support-only routing on validation subjects did not beat the best constant
  dev strategy under LOSO validation.

Therefore this CBraMod trial is the next step in the two-backbone one-dataset
plan, not a new scope expansion.

## Groups

- `NaiveAug_LS010`
- `ArtifactReject_LS010`
- `SoftWeight_noReject_LS010`
- `SASCert_SoftAR_LS010`

## First Step

Smoke only:

1. Resolve local CBraMod code and checkpoint.
2. Reuse the existing PhysioNetMI loader/cache without copying raw data.
3. Confirm one PhysioNetMI trial can pass through frozen CBraMod.
4. Confirm pooled feature shape and no NaN/Inf.

## Commands

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_cbramod_physionetmi_sascert_matched/01_smoke_cbramod_physionetmi.py \
  --device cuda --n-samples 8

PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_cbramod_physionetmi_sascert_matched/02_cache_cbramod_features.py \
  --device cuda --batch-size 32

PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_cbramod_physionetmi_sascert_matched/03_run_cbramod_mini_matrix.py \
  --device cuda \
  --targets 90 91 92 \
  --seeds 20 21 \
  --source-epochs 30 \
  --finetune-epochs 80 \
  --batch-size 64 \
  --feature-batch-size 32 \
  --output-tag mini
```

## Results

### Smoke

Status: `passed`

| Check | Result |
|---|---|
| PhysioNetMI cache reused | `/ai/224duibishiyan/615新研究/outputs/foundation_physio_mi_fullfinetune/data/physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz` |
| Cache shape | `[4917, 64, 640]` |
| CBraMod input shape | `[8, 64, 4, 200]` |
| CBraMod representation shape | `[8, 64, 4, 200]` |
| Pooled feature shape | `[8, 200]` |
| Checkpoint loaded keys | `211/211` |
| Missing/unexpected/shape mismatch | `0/0/0` |
| NaN/Inf count | `0/0` |
| Frozen backbone | `true` |
| Raw data copied | `false` |

Implementation note:

- The shared PhysioNetMI cache is canonical `[64,640]` at 160 Hz.
- CBraMod PhysioNet preprocessing expects 4 patches of 200 samples, so this
  smoke resamples each 4-second trial to 800 samples and reshapes to
  `[64,4,200]`.

Primary outputs:

- `outputs/cbramod_smoke_report.json`
- `outputs/cbramod_smoke_log.txt`
- `outputs/cbramod_smoke_run.log`

### Feature Cache

Status: `completed`

| Check | Result |
|---|---:|
| Feature cache shape | `[4917, 200]` |
| Labels shape | `[4917]` |
| Subjects range | `1-109` |
| Runs | `4,8,12` |
| Feature dtype | `float32` |
| NaN/Inf count | `0/0` |
| Raw trial array saved | `false` |
| Feature mean/std | `0.0035 / 0.1098` |

Primary outputs:

- `outputs/cbramod_original_features.npz`
- `outputs/cbramod_original_features_manifest.json`
- `outputs/cbramod_feature_cache_summary.json`
- `outputs/cbramod_feature_cache_run.log`

## Decision

`revise_cert_calibration`

### Mini Matrix

Status: `completed_diagnostic`

Scope:

- Targets: `90,91,92`
- Seeds: `20,21`
- Support: 5-shot per class
- Source epochs: `30`
- Fine-tune epochs: `80`
- Frozen CBraMod feature cache: `outputs/cbramod_original_features.npz`

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| `NaiveAug_LS010` | `0.5349` | `0.4983` | `0.0843` | `0.6931` |
| `ArtifactReject_LS010` | `0.5302` | `0.5001` | `0.0923` | `0.6931` |
| `SoftWeight_noReject_LS010` | `0.5360` | `0.4925` | `0.0866` | `0.6937` |
| `SASCert_SoftAR_LS010` | `0.5473` | `0.4988` | `0.1120` | `0.6932` |

SASCert SoftAR vs Naive:

- Balanced accuracy: `+1.24pp`
- Macro-F1: `+0.05pp`
- ECE: `+2.77pp`
- NLL: `+0.00018`
- Subject win rate by Macro-F1: `0.3333`
- Seed win rate by Macro-F1: `0.5000`

Interpretation:

- Frozen CBraMod on this PhysioNetMI few-shot protocol is much weaker than the
  source-tuned ST-EEGFormer branch.
- `SASCert_SoftAR_LS010` improves BAcc slightly but does not improve Macro-F1
  meaningfully and worsens calibration beyond the project gate.
- This mini result should not be expanded directly into a full confirmatory
  run. The next step is a bounded failure review of the CBraMod feature/training
  link rather than another method variant.

Primary outputs:

- `outputs/cbramod_physionetmi_sascert_metrics_mini.csv`
- `outputs/cbramod_physionetmi_paired_comparison_mini.csv`
- `outputs/compact_cbramod_physionetmi_sascert_result_mini.json`
- `outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_mini.md`
- `outputs/score_rows/mini/`

### Failure Review

Status: `completed`

Key facts:

- NoAug diagnostic: BAcc `0.5424`, Macro-F1 `0.5159`, ECE `0.0817`.
- Source-only target diagnostic: BAcc `0.5068`, Macro-F1 `0.5001`, ECE `0.0043`.
- Current SAS total score clean-vs-bad AUC: `0.1969`.
- Direction-fixed total score AUC: `0.8911`.
- Artifact-gate physio score AUC: `0.9022`.
- Current SAS total score direction is wrong on the mixed bad candidate pool.

Decision from review:

- `revise_cert_calibration`
- The first broken link is the current mixed-bad SAS score direction, not the
  existence of a useful cert signal.
- The frozen CBraMod feature/head link is weak, so wrong scoring cannot be
  rescued by training.

Primary outputs:

- `FAILURE_REVIEW.md`
- `failure_review_summary.json`
- `outputs/cbramod_physionetmi_noaug_diagnostic_mini.csv`

### Repaired Score Mini

Status: `completed_diagnostic`

Repaired score:

```text
score_artifact_gate_physio = physio_score, with artifact-risk p90 samples set to 0 before rank weighting
```

Scope:

- Targets: `90,91,92`
- Seeds: `20,21`
- Groups: `NoAug_LS010`, `NaiveAug_LS010`, `RepairedSoftWeight_artifact_gate_physio_LS010`

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| `NoAug_LS010` | `0.5231` | `0.4954` | `0.0595` | `0.6927` |
| `NaiveAug_LS010` | `0.5349` | `0.4983` | `0.0843` | `0.6931` |
| `RepairedSoftWeight_artifact_gate_physio_LS010` | `0.5461` | `0.5409` | `0.1070` | `0.6916` |

RepairedSoftWeight vs Naive:

- Balanced accuracy: `+1.11pp`
- Macro-F1: `+4.26pp`
- ECE: `+2.27pp`
- NLL: `-0.00148`
- Brier: `-0.00147`
- Subject win rate by Macro-F1: `0.6667`
- Seed win rate by Macro-F1: `1.0000`

Interpretation:

- Direction repair is meaningful: classification improves substantially on the
  mini matrix.
- Calibration remains outside the gate: ECE worsens by more than `+0.01`.
- Do not expand to full CBraMod PhysioNetMI yet. The next allowed step is a
  single calibration-aware repaired-score/loss design, not another broad method
  sweep.

Primary outputs:

- `outputs/cbramod_physionetmi_sascert_metrics_repaired_mini.csv`
- `outputs/cbramod_physionetmi_paired_comparison_repaired_mini.csv`
- `outputs/compact_cbramod_physionetmi_sascert_result_repaired_mini.json`
- `outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_repaired_mini.md`

### Calibration-Aware Repair Mini

Status: `completed_diagnostic_failed_calibration_gate`

Calibration design:

```text
Use the same repaired artifact-gate physio soft weighting.
After head training, fit one scalar temperature on target support only.
Apply that temperature to held-out target predictions.
No target held-out labels are used for fitting the temperature.
```

Scope:

- Targets: `90,91,92`
- Seeds: `20,21`
- Groups: `NoAug_LS010`, `NaiveAug_LS010`,
  `RepairedSoftWeight_artifact_gate_physio_LS010`,
  `RepairedSoftWeightTemp_artifact_gate_physio_LS010`

Temperature observation:

- Fitted temperature for all repaired-temp folds: `0.5`.
- This means support-only calibration sharpened predictions rather than
  softening them.

| Group | BAcc | Macro-F1 | ECE | NLL |
|---|---:|---:|---:|---:|
| `NoAug_LS010` | `0.5231` | `0.4954` | `0.0595` | `0.6927` |
| `NaiveAug_LS010` | `0.5349` | `0.4983` | `0.0843` | `0.6931` |
| `RepairedSoftWeight_artifact_gate_physio_LS010` | `0.5461` | `0.5409` | `0.1070` | `0.6916` |
| `RepairedSoftWeightTemp_artifact_gate_physio_LS010` | `0.5461` | `0.5409` | `0.1064` | `0.6940` |

Temp-calibrated repaired vs Naive:

- Balanced accuracy: `+1.11pp`
- Macro-F1: `+4.26pp`
- ECE: `+2.21pp`
- NLL: `+0.00094`
- Brier: `+0.00065`
- Subject win rate by Macro-F1: `0.6667`
- Seed win rate by Macro-F1: `1.0000`

Interpretation:

- Classification signal remains strong after score-direction repair.
- Simple support-only temperature scaling does not fix the calibration risk.
- Because ECE still violates the `+0.01` gate and NLL/Brier worsen relative to
  Naive, this branch should not be expanded to full targets.

Decision:

- `park_cbramod_physionetmi_full_expansion`
- Keep the repaired-score insight as evidence that SAS-Cert components can be
  backbone/dataset-direction dependent.
- Do not keep tuning calibration on the same mini result without a new
  hypothesis. The next mainline action should be a cross-backbone certificate
  direction audit or a return to ST-EEGFormer reliability, not another CBraMod
  full run.

Primary outputs:

- `outputs/cbramod_physionetmi_sascert_metrics_calibrated_mini.csv`
- `outputs/cbramod_physionetmi_paired_comparison_calibrated_mini.csv`
- `outputs/compact_cbramod_physionetmi_sascert_result_calibrated_mini.json`
- `outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_calibrated_mini.md`
