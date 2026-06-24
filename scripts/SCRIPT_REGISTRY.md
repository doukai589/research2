# Script Registry

This registry classifies scripts so the project does not become a flat pile of one-off entrypoints.

## Stable Utilities

| Script | Role |
|---|---|
| `manage_trial.py` | Create isolated `workbench/` trial boxes. |

## Current Active Experimental Entrypoints

| Script | Status | Notes |
|---|---|---|
| `40_run_physio_mi_foundation_fullfinetune.py` | active reference | Shared PhysioNetMI full fine-tune for ST-EEGFormer, LaBraM, EEGPT. Keep as reference until its logic is promoted into `sas_core/`. |

## Setup / Audit Scripts

| Script | Status | Notes |
|---|---|---|
| `00_setup_backbones_and_datasets.py` | legacy setup | Used for initial backbone/dataset setup audit. |
| `01_inventory_local_datasets.py` | legacy setup | Dataset inventory. |
| `02_summarize_setup_audit.py` | legacy setup | Summary for setup audit. |
| `03_smoke_backbone_dataset_compatibility.py` | legacy setup | Initial backbone/dataset compatibility smoke. |
| `04_summarize_step1_compatibility.py` | legacy setup | Summary for step1 compatibility. |

## Reproduction / Parked Branch Scripts

| Script | Status | Notes |
|---|---|---|
| `20_extract_paper_targets.py` | parked | Paper target extraction. |
| `21_reproduce_cbramod_paper.py` | parked/reference | CBraMod paper-code reproduction. |
| `22_prepare_mirepnet_adapter.py` | parked | MIRepNet adapter attempt. |
| `23_reproduce_mirepnet_paper.py` | parked | MIRepNet reproduction attempt. |
| `24_summarize_paper_reproduction.py` | parked | Paper reproduction summary. |
| `25_repair_repro_gate.py` | parked | Reproduction gate repair. |
| `26_prepare_mirepnet_bnci2014001_adapter.py` | parked | MIRepNet BNCI adapter. |
| `27_prepare_mirepnet_bnci2014001_moabb_session_t_adapter.py` | parked | Corrected MIRepNet MOABB/session-T adapter. |

## Policy

- New ideas should start in `workbench/`.
- Promote reusable logic into `sas_core/`.
- Keep stable entrypoints small.
- Do not keep adding numbered scripts indefinitely.

