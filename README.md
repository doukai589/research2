# SAS-Cert EEG Research Workspace

This workspace is organized for iterative EEG foundation-model experiments.

Core rule:

```text
third_party is kept close to upstream code.
workbench is for fast trials.
sas_core is for reusable project code after a trial proves useful.
outputs/runs is for auditable experiment results.
archive is for parked or failed branches.
```

Use `PROJECT_MANAGEMENT.md` as the long-running human project ledger.

## Main Directories

- `third_party/`: external repositories and backbone code.
- `sas_core/`: reusable in-project modules for datasets, backbones, augmentation, certification, training, metrics, and reporting.
- `configs/experiments/`: stable experiment YAML/config files.
- `scripts/`: small entrypoints and project utilities.
- `workbench/`: temporary trial boxes. New ideas start here.
- `outputs/runs/`: normalized long-term run outputs.
- `outputs/`: historical and active experiment outputs.
- `archive/`: failed, parked, or legacy branches with notes.
- `artifacts/source_archives/`: downloaded zip/source archives and checkpoint bundles.
- `docs/references/`: papers and reference material.

## Current Route

The current main line is:

1. CBraMod + BCIC-IV-2a as the completed SAS-Cert anchor.
2. ST-EEGFormer-small + PhysioNetMI as the first companion foundation backbone.
3. LaBraM as secondary reliability/calibration baseline.
4. EEGPT and MIRepNet are paused outside the main line.

# research2
