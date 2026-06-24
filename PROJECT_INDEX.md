# Project Index

This file maps the current project layout. It is intentionally path-stable: experiment artifacts are indexed before they are moved.

## Root Files

- `README.md`: current workspace overview.
- `PROJECT_MANAGEMENT.md`: long-term project ledger and current decisions.
- `PROJECT_INDEX.md`: this directory map.
- `RUN_REGISTRY.md`: durable run index.
- `PATCHES.md`: third-party/runtime patch log.
- `.gitignore`: excludes outputs, weights, raw data, archives, and local runtime state.

## Main Directories

- `scripts/`: stable utilities plus historical setup/reproduction scripts; see `scripts/SCRIPT_REGISTRY.md`.
- `workbench/`: temporary trial boxes for fast iteration.
- `sas_core/`: reusable code promoted from successful workbench trials.
- `configs/experiments/`: stable experiment configs.
- `archive/`: parked, failed, or legacy branches.
- `artifacts/source_archives/`: zip/checkpoint/source archives moved out of the root.
- `docs/references/`: papers and reference materials.
- `outputs/`: root-level reports and outputs for setup audits, paper reproduction, and MIRepNet runs.
- `outputs/runs/`: normalized output location for new durable runs.
- `third_party/`: local third-party backbone code and copied/extracted projects.
- `sas_cert_mve/`: original EEGNet-style SAS-Cert MVE code.
- `sas_cert_mve_outputs/`: original EEGNet-style SAS-Cert MVE outputs and final analysis.
- `sas_cert_cbramod_mve/`: independent SAS-Cert-CBraMod MVE project with its own scripts, code, configs, and outputs.
- `显存占位/`: GPU memory-holder utility; do not start it during active training runs unless intentionally reserving GPU.

## Important Output Reports

- `sas_cert_mve_outputs/SAS_CERT_MVE_FULL_OUTPUT_AND_ANALYSIS.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_MVE_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V2_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V3_CALIBRATION_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V4_CONFIRMATORY_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V5_LOCKED_CONFIRMATORY_REPORT.md`
- `outputs/paper_code_runs/PAPER_CODE_RUN_REPORT.md`
- `outputs/mirepnet_full_paper_code/MIRepNet_FULL_PAPER_CODE_REPORT.md`
- `outputs/mirepnet_full_paper_code/MIRepNet_MOABB_SESSION_T_RERUN_REPORT.md`
- `outputs/foundation_physio_mi_fullfinetune/PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md`

## Reference / Archive Files

- `artifacts/source_archives/CBraMod-main.zip`
- `artifacts/source_archives/25866970.zip`
- `docs/references/Wang 等 - 2025 - CBraMod A Criss-Cross Brain Foundation Model for EEG Decoding.pdf`
- `docs/references/Liu 等 - 2026 - MIRepNet A pipeline and pre-trained model for EEG-based motor imagery classification.pdf`

## Data Handling Notes

- Raw datasets are referenced from existing locations and should not be copied into experiment project folders.
- MIRepNet processed adapter files currently live in `third_party/backbones/MIRepNet/data/BNCI2014001`.
- Previous MIRepNet adapter backups are under `outputs/mirepnet_full_paper_code/adapter_backups`.
- Feature caches and generated scores should remain under their experiment output directories with manifest files.

## Workflow Notes

- New ideas start in `workbench/`.
- Reusable code is promoted into `sas_core/`.
- Stable experiment choices live in `configs/experiments/`.
- New durable outputs should use `outputs/runs/<YYYYMMDD>_<short_name>/`.
- Do not add endless numbered one-off scripts; classify them in `scripts/SCRIPT_REGISTRY.md`.
