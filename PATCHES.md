# Patch Log

Record any direct or runtime patch to third-party code here.

## Current Known Runtime Patches

### EEGPT PhysioNetMI Runner Compatibility

- Location: `scripts/40_run_physio_mi_foundation_fullfinetune.py`
- Scope: runner-local monkey patches only; no third-party source file was modified.
- Reason:
  - Released EEGPT downstream code uses constrained layers with autocast behavior incompatible with the current torch runtime.
  - Released EEGPT downstream code calls PyTorch-2 `scaled_dot_product_attention`, which is unavailable in the current torch runtime.
- Effect:
  - Constrained Conv1d/Linear layers are forced to float32 in the runner.
  - Attention uses a legacy explicit softmax attention implementation.

