# MIRepNet Paper Reproduction Report

## Adapter

- adapter_ready: `False`
- reason: not_generated_until_validated_inverse_distance_interpolation_and_EA_are_implemented

## Protocol Separation

### code_exact
- BNCI2014001: `code_not_released_for_dataset`; entrypoint `/ai/224duibishiyan/615新研究/third_party/backbones/MIRepNet/finetune.py`; README command supports BNCI2014004; README says code for other datasets will be released.
- BNCI2014001-4: `code_not_released_for_dataset`; entrypoint `/ai/224duibishiyan/615新研究/third_party/backbones/MIRepNet/finetune.py`; Repository dataset.py has BNCI2014001-4 branch but README says other dataset code to be released; dependency/preprocessed files missing.

### paper_exact
- BNCI2014001: `needs_adapter_and_dependencies`; entrypoint `paper 45-channel template + EA + 30% target fine-tune`; Requires adapter to X.npy/labels.npy and local dependencies wandb/einops.
- BNCI2014001-4: `needs_adapter_and_dependencies`; entrypoint `paper 45-channel template + EA + 30% target fine-tune`; Requires validated inverse-distance channel interpolation and EA behavior.

### executable_hybrid
- BNCI2014001: `not_ready`; entrypoint `local adapter audit only`; Hybrid cannot be scored until adapter is validated.
- BNCI2014001-4: `not_ready`; entrypoint `local adapter audit only`; Hybrid cannot be scored until adapter is validated.

## Results

| Dataset | Protocol | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BNCI2014001 | code_exact | official_code_default | not_run |  |  | failed | official_repo_behavior_unclear |
| BNCI2014001 | paper_exact | likely 2-class left/right | accuracy_full_finetune | 0.8177±0.0027 |  | failed | adapter_needed |
| BNCI2014001 | paper_exact | likely 2-class left/right | accuracy_linear_probe | 0.7536±0.0226 |  | failed | adapter_needed |
| BNCI2014001 | executable_hybrid | adapter_hybrid | not_run |  |  | failed | adapter_needed |
| BNCI2014001-4 | code_exact | official_code_default | not_run |  |  | failed | official_repo_behavior_unclear |
| BNCI2014001-4 | paper_exact | 4-class full | accuracy_full_finetune | 0.6414±0.0031 |  | failed | adapter_needed |
| BNCI2014001-4 | paper_exact | 4-class full | accuracy_linear_probe | 0.5194±0.0184 |  | failed | adapter_needed |
| BNCI2014001-4 | executable_hybrid | adapter_hybrid | not_run |  |  | failed | adapter_needed |
