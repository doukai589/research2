# sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi

## Intent

Test whether v1.3's weak subject/seed stability comes from global rank
normalization or from an overly clean ordinary augmentation pool.

## Protocol

- Backbone: ST-EEGFormer-small.
- Dataset: PhysioNetMI / EEGMMI left-vs-right motor imagery.
- Runs: R04/R08/R12.
- Targets: 90-109.
- Seeds: 20-24.
- Source-tuned checkpoint:
  `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`.

## Regular Pool Experiment

Groups:

- `NaiveAug_LS010`
- `SAS-Cert-CU-LS-v1.3`
- `SAS-Cert-SCB-CU-LS-v1.4`

v1.4 uses only content utility:

- `E_embed`
- `E_proto`

It does not use artifact / physio / style / prediction consistency for training
weights.

## Risk-Mixed Pool Experiment

Groups:

- `RiskMixed_NaiveAug_LS010`
- `RiskMixed_SAS-Cert-SCB-CU-LS-v1.4`

Candidate pool:

- 70% mild augmentation
- 30% risky augmentation

Risky augmentations:

- `strong_frequency_mask`
- `strong_channel_dropout`
- `emg_like_burst`
- `eog_like_drift`
- `covariance_perturbation`

## Commands

```bash
python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --experiment v1_4_regular \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_4_regular \
  --output-dir workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs

python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py \
  --experiment v1_4_riskmixed \
  --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \
  --seeds 20 21 22 23 24 \
  --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt \
  --feature-tag st_source_tuned_seed3407 \
  --output-tag v1_4_riskmixed \
  --output-dir workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs
```

## Required Outputs

- `outputs/SASCERT_V1_4_SCB_CU_RISKMIXED_REPORT.md`
- `outputs/compact_sascert_v1_4_result.json`
- `outputs/metrics_v1_4_regular_pool.csv`
- `outputs/paired_comparison_v1_4_regular_pool.csv`
- `outputs/metrics_v1_4_riskmixed_pool.csv`
- `outputs/paired_comparison_v1_4_riskmixed_pool.csv`
- `outputs/per_subject_delta_table.csv`
- `outputs/per_class_delta_table.csv`
- `outputs/per_subject_component_corr.csv`
- `outputs/weight_distribution_by_subject_class.csv`
- `outputs/riskmixed_diagnostic_summary.csv`
- `outputs/leakage_audit_v1_4.json`

## Results

- Regular pool v1.4 vs `NaiveAug_LS010`:
  - delta BAcc `+0.000626`
  - delta Macro-F1 `+0.000295`
  - delta ECE `-0.001412`
  - delta NLL `-0.001908`
  - delta Brier `-0.000977`
- Regular pool v1.4 vs v1.3:
  - delta BAcc `-0.001453`
  - delta Macro-F1 `-0.001640`
  - subject win rate Macro-F1 `0.00`
- Risk-mixed pool v1.4 vs `RiskMixed_NaiveAug_LS010`:
  - delta BAcc `+0.000476`
  - delta Macro-F1 `+0.000450`
  - delta ECE `-0.001121`
  - delta NLL `-0.003801`
  - delta Brier `-0.002236`
- Localization audit:
  - `E_proto` remains the strongest utility signal.
  - Subject/class ranknorm removed mean weight imbalance:
    - all subject/class bins have mean v1.4 weight `1.0`.
  - Subject win rate remained weak, so global ranknorm unfairness is not the
    main failure mode.
- Leakage audit:
  - `passed`
  - target test was used only for final evaluation.

## Decision

- `limit_training_use_to_diagnostic_or_riskmixed`
- Do not enter CBraMod recheck from v1.4.
- Next branch should either continue repairing subject-balanced utility or keep
  SAS-Cert training use limited to explicit risk-mixed augmentation settings.
