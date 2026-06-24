# Reference Algorithm Audit Report

Task: `SASCERT_DIRECT_USE_ALGORITHM_AUDIT_AND_V1_BLUEPRINT`

## Scope

This audit downloaded and inspected only whitelisted reference projects for
SAS-Cert-SoftAR-LS v1.1. It did not train any model, run a SAS-Cert experiment,
copy raw EEG data, or modify old experimental outputs. Installation checks used
an isolated `pip --target` directory:

`/ai/224duibishiyan/615新研究/outputs/reference_algorithm_audit/python_target`

Third-party reference code was placed under:

`/ai/224duibishiyan/615新研究/third_party/reference_algorithms`

Reports were written under:

`/ai/224duibishiyan/615新研究/outputs/reference_algorithm_audit`

## Summary By Recommendation

### use_now

- pyRiemann
- MNE-Features

### use_offline_only

- Autoreject

### use_later

- EEG-DLite
- MOABB
- Channel Reflection

### cite_only

- None

### blocked

- MNE-ICALabel
- Braindecode

## Inventory

            | Project | Install | Import | Smoke | Recommendation | Useful for |
            |---|---|---|---|---|---|
| MNE-ICALabel | pip_install_no_deps_ok:mne-icalabel | import_failed | smoke_failed | blocked | artifact_probability/safety_gate |
| Autoreject | pip_install_no_deps_ok:autoreject h5io | import_ok | smoke_ok | use_offline_only | safety_gate |
| pyRiemann | pip_install_no_deps_ok:pyriemann | import_ok | smoke_ok | use_now | physio_score/style_score |
| MNE-Features | pip_install_no_deps_ok:mne-features PyWavelets | import_ok | smoke_ok | use_now | safety_gate/physio_score/style_score |
| Braindecode | pip_install_no_deps_ok:braindecode skorch | import_failed | smoke_failed | blocked | augmentation_pool/baseline |
| EEG-DLite | not_applicable | not_applicable | not_applicable | use_later | augmentation_pool/redundancy_filtering |
| MOABB | pip_install_no_deps_ok:moabb | import_ok | smoke_ok | use_later | benchmark |
| Channel Reflection | not_applicable | not_applicable | not_applicable | use_later | augmentation_pool |

## Blocked Details

- MNE-ICALabel: package body installed into the isolated target, but import
  failed because the current MNE package exposed in this environment does not
  provide `mne.io.Info`, which the installed MNE-ICALabel expects. Do not force
  it into v1.1 without a version-matched MNE/ICALabel environment.
- Braindecode: package body and `skorch` installed into the isolated target,
  but import failed at a missing transitive dependency (`tabulate`) and the
  cloned package declares a newer MNE/BIDS/WFDB-heavy stack. Treat it as
  `use_later` conceptually, but `blocked` for direct current-environment use.

## Channel Reflection Source Note

Search found `https://github.com/wzwvv/EEGAug`, whose README states
it is the official implementation of Channel Reflection. The arXiv
HTML v1 page also footnotes `https://github.com/sylyoung/DeepTransferEEG`.
This audit used the README-identified `wzwvv/EEGAug` repository and
performed a sparse clone excluding `data/` to avoid copying EEG data.

## Interpretation

`pyRiemann` and `MNE-Features`, when smoke checks pass, are the
strongest direct v1.1 dependencies because they operate on synthetic
trial tensors or covariance/features without requiring dataset
loaders. `Autoreject` and `MNE-ICALabel` should be treated as
offline safety/audit tools first because they require MNE Epochs/Raw,
channel metadata, and in the ICLabel case a fitted ICA. `Braindecode`
is useful for augmentation and baselines but should not determine the
SAS-Cert logic. `EEG-DLite`, `MOABB`, and Channel Reflection are
useful references or later candidates, not required v1.1 components.
