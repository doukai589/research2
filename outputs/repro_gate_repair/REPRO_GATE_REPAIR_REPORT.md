# Reproduction Gate Repair Report

## CBraMod
- code_exact_ready: `False`
- dry_run_success: `False`
- dry_run_status: `blocked`
- paper_exact_preprocessing_ready: `True`
- needs_lmdb_generation: `True`
- code-paper gap policy: if exact LMDB follows paper preprocessing/split, official code should be primary; otherwise exact claim remains blocked.

## MIRepNet
- dependencies_missing: `['wandb', 'einops']`
- adapter_status: `adapter_preprocessing_ready`
- tiny_x_shape: `[8, 45, 1000]`
- forward_success: `True`
- forward_status: `adapter_smoke_ready`

## Decision
- decision: `MIREPNET_ADAPTER_NEXT`
- next_action: Package MIRepNet full adapter datasets and then run controlled reproduction.
