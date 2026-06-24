# Output Index

This directory stores root-level experiment outputs. Do not delete or rename these directories unless the corresponding reports have been archived.

New durable outputs should preferably use `outputs/runs/<YYYYMMDD>_<short_name>/`. Historical output roots are kept in place to preserve reproducibility.

## Directories

- `setup_audit/`: initial backbone and dataset inventory.
- `setup_audit_step1/`: backbone/dataset smoke and compatibility reports.
- `paper_reproduction/`: code/paper/hybrid reproduction-gate tables and early reports.
- `repro_gate_repair/`: repaired reproduction-gate audit outputs.
- `paper_code_runs/`: CBraMod paper-code execution on existing processed LMDBs.
- `mirepnet_full_paper_code/`: MIRepNet adapter manifests, logs, metrics, and rerun reports.
- `foundation_physio_mi_fullfinetune/`: shared PhysioNetMI full fine-tune run for ST-EEGFormer, LaBraM, and EEGPT.
- `runs/`: normalized output root for future durable runs.

## Key Reports

- `setup_audit/SETUP_AUDIT_REPORT.md`
- `setup_audit_step1/STEP1_COMPATIBILITY_REPORT.md`
- `paper_reproduction/PAPER_REPRODUCTION_GATE_REPORT.md`
- `repro_gate_repair/REPRO_GATE_REPAIR_REPORT.md`
- `paper_code_runs/PAPER_CODE_RUN_REPORT.md`
- `mirepnet_full_paper_code/MIRepNet_FULL_PAPER_CODE_REPORT.md`
- `mirepnet_full_paper_code/MIRepNet_MOABB_SESSION_T_RERUN_REPORT.md`
- `foundation_physio_mi_fullfinetune/PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md`

## Current Foundation PhysioNetMI Snapshot

- ST-EEGFormer-small: BAcc `76.69%`, Macro-F1 `76.35%`.
- LaBraM-base: BAcc `65.47%`, Macro-F1 `64.81%`.
- EEGPT-large4E: BAcc `52.11%`, Macro-F1 `52.09%`.
- Interpretation: ST-EEGFormer-small is the first companion backbone beyond CBraMod; LaBraM is secondary; EEGPT is paused.

## Current MIRepNet Result Snapshot

- Dataset: `BNCI2014001-4`
- Corrected adapter: MOABB `0train` only, shape `[2592, 22, 1000]`.
- Best rerun: batch8 default-like, mean accuracy `60.23%`.
- Paper target: `64.14%`.
- Gap after correction: `-3.91pp`.
