# SAS Core

Reusable project modules live here after they have proven useful in `workbench/`.

Suggested module boundaries:

- `data/`: dataset loaders and split protocols.
- `backbones/`: wrappers for CBraMod, ST-EEGFormer, LaBraM, EEGPT, etc.
- `augmentation/`: clean/bad augmentation operators.
- `cert/`: SAS-Cert scores, rank normalization, artifact gates.
- `training/`: training loops and loss utilities.
- `metrics/`: accuracy, F1, kappa, ECE, NLL, Brier.
- `reporting/`: summary/report helpers.
- `utils/`: seed, IO, path, logging helpers.

Do not dump one-off runners here. Start them in `workbench/` first.

