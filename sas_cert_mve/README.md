# SAS-Cert-EEG MVE

This folder implements the three-layer minimal validation experiment for
BCI Competition IV-2a:

1. Layer 1: test whether controlled bad augmentations hurt LOSO few-shot EEG adaptation.
2. Layer 2: test whether SAS-Cert-Lite ranks clean augmentations above bad ones.
3. Layer 3: compare NoAug, NaiveAug, Random50, and SASCertTop50 fine-tuning.

The implementation is intentionally independent from the existing CBraMod and
baseline folders. It reads the existing BCIC IV-2a `.mat` files from:

```bash
../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014
```

## Environment

The default Python in this workspace may not include the scientific stack. Install
the local dependencies in the environment you intend to run:

```bash
python -m pip install -r sas_cert_mve/requirements.txt
```

PyTorch is also required; the current workspace already has it installed.

## Smoke Test

```bash
python -m sas_cert_mve.run_three_layers --smoke --source-epochs 1 --finetune-epochs 1
```

This loads A01, verifies `[22, 800]` samples and T/E separation, then runs one
target subject, one seed, and one training epoch.

## Full Three-Layer Run

```bash
python -m sas_cert_mve.run_three_layers \
  --data-root ../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014 \
  --out-dir sas_cert_mve_outputs/full \
  --subjects 1,2,3,4,5,6,7,8,9 \
  --seeds 20,21,22 \
  --shot 5 \
  --aug-per-trial 5 \
  --source-epochs 25 \
  --finetune-epochs 20
```

Outputs:

- `layer1_metrics.csv`
- `layer1_summary.json`
- `cert_scores.csv`
- `layer2_summary.json`
- `layer3_metrics.csv`
- `layer3_summary.json`

