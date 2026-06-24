# Script Index

Root-level scripts are grouped by experiment stage.

## Setup And Inventory

- `00_setup_backbones_and_datasets.py`: create/verify local backbone and dataset setup.
- `01_inventory_local_datasets.py`: inventory local EEG datasets.
- `02_summarize_setup_audit.py`: summarize setup and dataset audit.
- `03_smoke_backbone_dataset_compatibility.py`: smoke test backbone/dataset compatibility.
- `04_summarize_step1_compatibility.py`: summarize compatibility results.

## Paper Reproduction Gate

- `20_extract_paper_targets.py`: extract target paper metrics.
- `21_reproduce_cbramod_paper.py`: run CBraMod paper-code style reproduction.
- `22_prepare_mirepnet_adapter.py`: earlier MIRepNet adapter preparation.
- `23_reproduce_mirepnet_paper.py`: MIRepNet reproduction runner.
- `24_summarize_paper_reproduction.py`: summarize paper reproduction attempts.
- `25_repair_repro_gate.py`: repair and validate reproduction gate issues.

## MIRepNet BNCI2014001-4 Adapter Runs

- `26_prepare_mirepnet_bnci2014001_adapter.py`: raw `.mat` T/E adapter; now considered hybrid/diagnostic.
- `27_prepare_mirepnet_bnci2014001_moabb_session_t_adapter.py`: MOABB `0train` only adapter aligned with author issue guidance.

## Current Guidance

- Prefer `27_prepare_mirepnet_bnci2014001_moabb_session_t_adapter.py` for BNCI2014001-4 reproduction checks.
- Treat outputs from the raw `.mat` T+E adapter as diagnostic, not paper-exact.
- Keep script numbering stable; add new scripts with the next number instead of renaming old scripts.
# Scripts

See `scripts/SCRIPT_REGISTRY.md` for the current classification of scripts.

New experimental ideas should start in `workbench/` via:

```bash
python scripts/manage_trial.py new <trial_name>
```

Only promote a runner into `scripts/` after it has become a stable entrypoint.

