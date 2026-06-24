# Backbone Download Audit Report

## Summary

Status: **partial**. ST-EEGFormer and LaBraM are ready for smoke tests. EEGPT code is present, but the official Figshare checkpoint could not be downloaded programmatically because the Figshare share page returned a WAF challenge.

## ST-EEGFormer

- Code path: `/ai/224duibishiyan/615新研究/third_party/backbones/STEEGFormer`
- Commit: `542ee17918c3c2c36ba1d4ea02bedff5eb149370`
- Weights:
  - small: `/ai/224duibishiyan/615新研究/third_party/backbones/STEEGFormer/pretrained_weights/ST-EEGFormer-small/checkpoint-300.pth` (0.367 GiB)
  - base: `/ai/224duibishiyan/615新研究/third_party/backbones/STEEGFormer/pretrained_weights/ST-EEGFormer-base/checkpoint-288.pth` (1.242 GiB)
  - large: `/ai/224duibishiyan/615新研究/third_party/backbones/STEEGFormer/pretrained_weights/ST-EEGFormer-large/large_weights_only_196.pth` (1.226 GiB)
  - largeV2: recorded only, not downloaded for first smoke
- README input hint: 128 Hz EEG, <=6 s segments, up to 142 channels.
- Tutorials found:
  - `easy_start/simple_example.ipynb`
  - `easy_start/bci_iv2a_dataset_tutorial.ipynb`
- Recommended first smoke: small, then base.

## LaBraM

- Code path: `/ai/224duibishiyan/615新研究/third_party/backbones/LaBraM`
- Commit: `c431221e6cfd23dbfa9950e0180682fb322b0548`
- `checkpoints/labram-base.pth`: True (92.1 MiB)
- `checkpoints/vqnsp.pth`: True (90.4 MiB)
- Git LFS pointer detected: false for both audited checkpoint files.
- README input hints: provide channel order list, resample to 200 Hz, use `labram_base_patch200_200` for downstream fine-tuning.

## EEGPT

- Code path: `/ai/224duibishiyan/615新研究/third_party/backbones/EEGPT`
- Commit: `a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705`
- Expected checkpoint: `/ai/224duibishiyan/615新研究/third_party/backbones/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
- Checkpoint exists: False
- Official source: `https://figshare.com/s/e37df4f8a907a866df4b`
- Download result: manual download required. Direct request returned HTTP 202 with `x-amzn-waf-action: challenge`.
- Expected input: 58 channels, 256 Hz, 4 s, patch size 64.
- Loader path exists: `downstream/Modules/models/EEGPT_mcae_finetune.py`.

## Output Files

- `/ai/224duibishiyan/615新研究/outputs/backbone_download_audit/backbone_download_inventory.csv`
- `/ai/224duibishiyan/615新研究/outputs/backbone_download_audit/backbone_download_inventory.json`
- `/ai/224duibishiyan/615新研究/outputs/backbone_download_audit/compact_backbone_download_result.json`

## Next Action

1. Run ST-EEGFormer-small smoke first.
2. Run LaBraM smoke after channel-order adapter check.
3. Keep EEGPT deferred until the official Figshare checkpoint is manually available.
