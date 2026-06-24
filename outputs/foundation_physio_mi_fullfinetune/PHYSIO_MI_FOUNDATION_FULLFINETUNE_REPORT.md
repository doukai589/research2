# PhysioNetMI Foundation Backbone Full Fine-tuning Report

- Protocol: `paper_aligned_common_protocol`
- Dataset source: `/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files`
- Raw data copied: `false`
- Task: left/right motor imagery, runs `[4, 8, 12]`
- Split: subjects 1-70 train, 71-89 val, 90-109 test
- Epochs requested: `30`
- Seed: `3407`

## Results

| Model | Best Epoch | Test Acc | Test BAcc | Macro-F1 | Kappa | AUROC | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| st_eegformer_small | 11 | 0.7667 | 0.7669 | 0.7635 | 0.5336 | 0.8712 | 0.9054 | 0.1816 |
| labram_base | 11 | 0.6544 | 0.6547 | 0.6481 | 0.3093 | 0.7122 | 0.6453 | 0.0620 |
| eegpt_large4e | 20 | 0.5211 | 0.5211 | 0.5209 | 0.0421 | 0.5384 | 2.8741 | 0.4614 |

## Notes

- This is a shared executable PhysioNetMI protocol, not an official code-exact reproduction for any single backbone repository.
- Target test subjects are used only for final evaluation.
- Derived cache stores processed trial windows, not raw EDF files.
- ST-EEGFormer-small and LaBraM-base train all parameters. EEGPT-large4E trains all core parameters; 16 static rotary-frequency scalars remain frozen to avoid cached-graph backward failure in the released EEGPT module.
- EEGPT required two runner-local compatibility patches: disable autocast in its constrained Conv1d/Linear layers and replace PyTorch-2 `scaled_dot_product_attention` with a legacy attention implementation for the current torch runtime.

## Interpretation

- ST-EEGFormer-small is the clear winner in this run: test balanced accuracy `0.7669`, macro-F1 `0.7635`, AUROC `0.8712`.
- LaBraM-base is usable but weaker: test balanced accuracy `0.6547`, macro-F1 `0.6481`.
- EEGPT-large4E performs near chance under this protocol: test balanced accuracy `0.5211`, macro-F1 `0.5209`, with very poor calibration (`ECE=0.4614`, `NLL=2.8741`).
- Compared with the earlier CBraMod PhysioNetMI paper-code reference around balanced accuracy `0.6285`, ST-EEGFormer-small is substantially higher, LaBraM-base is modestly higher, and EEGPT-large4E is not competitive in the current adapter.
