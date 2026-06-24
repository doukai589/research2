# MIRepNet Full Paper-Code Run

## Scope

This run solved the missing MIRepNet `X.npy/labels.npy` input problem for BCIC IV-2a / BNCI2014001 and ran the MIRepNet 4-class full-finetune path.

Protocol label: `executable_hybrid_gpu_batch64_subject_filtered_adapter`.

This is not strict `code_exact` because the official README only documents `BNCI2014004` and says code for other datasets will be released. It is also not strict `paper_exact` because the paper pipeline had to be reconstructed locally from the available raw BCIC IV-2a `.mat` files.

## Data Adapter

Generated processed arrays:

- `third_party/backbones/MIRepNet/data/BNCI2014001/X.npy`
- `third_party/backbones/MIRepNet/data/BNCI2014001/labels.npy`
- `third_party/backbones/MIRepNet/data/BNCI2014001/subjects.npy`
- `third_party/backbones/MIRepNet/data/BNCI2014001/sessions.npy`
- `third_party/backbones/MIRepNet/data/BNCI2014001/trial_ids.npy`

Adapter details:

- Raw source: `/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014`
- Shape: `[5088, 22, 1000]`
- Classes: 4, balanced at 1272 trials per class
- Window: event + 2s to event + 6s
- Sampling rate: 250Hz
- Bandpass: 8-30Hz
- Raw dataset copied: no

The saved `X.npy` intentionally stays at 22 channels. MIRepNet official `process_and_replace_loader()` then applies EA and inverse-distance 45-channel interpolation after each train/validation split.

## Code Repairs

Small repairs were required to make the unreleased BNCI2014001-4 path executable:

- `dataset.py`: added optional `subjects.npy` filtering for `BNCI2014001` and `BNCI2014001-4`, otherwise every subject loop reused the full dataset.
- `dataset.py`: removed an unused circular import from `utils.utils`.
- `model/mlm.py`: added `map_location='cpu'` to checkpoint loading for CPU/GPU portability.

GPU was initially blocked by `显存占位/hold_gpu_mem.py`; that tmux session was stopped to free the GPU.

## Result

Official result CSV:

`third_party/backbones/MIRepNet/result/acc/BNCI2014001-4_MIRepNet_2026-06-22_15-28-12_results.csv`

| Subject | Accuracy |
| --- | ---: |
| 0 | 71.04% |
| 1 | 51.73% |
| 2 | 78.47% |
| 3 | 49.40% |
| 4 | 39.85% |
| 5 | 35.40% |
| 6 | 70.79% |
| 7 | 65.10% |
| 8 | 63.61% |
| Mean | 58.38% |

Paper target for `BNCI2014001-4` full finetune: `64.14% ± 0.31%`.

Delta: `-5.76pp`, not within 2 paper std.

## Interpretation

MIRepNet now runs end-to-end on the repaired BNCI2014001-4 adapter, but this executable run does not reproduce the paper result. The gap is meaningful: mean accuracy is 58.38% versus the paper's 64.14%.

The weakest subjects are 4 and 5, at 39.85% and 35.40%, which dominate the mean drop. Subjects 0, 2, and 6 are strong, so the model is not globally broken; the issue is subject robustness and/or a protocol mismatch.

Because this run required a local adapter and used batch size 64 instead of the official default 8 for feasible execution, it should be treated as `executable_hybrid`, not exact reproduction.

## Output Files

- Adapter manifest: `outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_adapter_manifest.json`
- Metrics CSV: `outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_4class_metrics.csv`
- Compact JSON: `outputs/mirepnet_full_paper_code/compact_mirepnet_full_result.json`
- GPU stdout: `outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_4class_full_gpu_batch64_stdout.log`
