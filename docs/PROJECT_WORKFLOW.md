# Project Workflow

This project uses a three-zone workflow.

## 1. External Code Zone

Path: `third_party/`

Purpose: keep upstream repositories and official weights/code.

Rules:

- Avoid editing third-party source directly.
- Prefer wrappers, adapters, or runner-local monkey patches.
- If third-party code must be modified, record it in `PATCHES.md`.

## 2. Workbench Zone

Path: `workbench/<date>_<trial_name>/`

Purpose: fast iteration without polluting the main project.

Each trial box should contain:

- `TRIAL.md`: intent, protocol, status, command, outputs.
- `config.yaml`: trial configuration when applicable.
- `runner.py` or small local utilities only when necessary.
- `outputs/`: temporary local outputs.

Promotion rule:

- If the trial succeeds and will be reused, move the useful code into `sas_core/` or a stable `scripts/` entrypoint.
- If the trial fails, summarize it and move or mark it under `archive/failed_trials/`.

## 3. Locked Core Zone

Paths:

- `sas_core/`
- `configs/experiments/`
- stable `scripts/`

Purpose: reusable code and repeatable experiments.

Rules:

- Do not add one-off code here until it has proven useful in workbench.
- Keep entrypoints small.
- Put reusable logic in `sas_core/`.
- Put experiment choices in config files.

## Run Output Rule

New durable runs should use:

```text
outputs/runs/<YYYYMMDD>_<short_name>/
  config.yaml
  metrics.csv
  summary.json
  report.md
  logs/
  checkpoints/
```

Historical outputs are left in place unless moving them would not break references.

## Version-Control Rule

Track code and management files. Do not track large outputs, weights, caches, raw EEG data, or archives.

