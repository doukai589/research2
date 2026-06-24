# Step0 Backbone And Dataset Inventory

## Paths

- workspace_root: `/ai/224duibishiyan/615新研究`
- project_root: `/ai/224duibishiyan/615新研究`

## Backbone Inventory

| Model | Status | Has weights | Commit | Local path | Notes |
| --- | --- | --- | --- | --- | --- |
| CBraMod | ready | True | `` | `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/third_party/CBraMod` | Code inspected. |
| MIRepNet | ready | True | `2bf16bcea493` | `/ai/224duibishiyan/615新研究/third_party/backbones/MIRepNet` | README mentions download. README mentions pretrain. README mentions channel. README mentions sampling. README mentions dataset. |
| LaBraM | ready | True | `c431221e6cfd` | `/ai/224duibishiyan/615新研究/third_party/backbones/LaBraM` | Weight download is README/manual if no checkpoint exists. Check channel configuration before downstream use. README mentions checkpoint. README mentions pretrain. README mentions c |
| EEGPT | code_only | False | `a0e0a8fad729` | `/ai/224duibishiyan/615新研究/third_party/backbones/EEGPT` | Review README for sampling rate/channel/patch constraints before use. README mentions download. README mentions checkpoint. README mentions pretrain. README mentions channel. READM |
| MFrFM | code_only | False | `aab96df836e8` | `/ai/224duibishiyan/615新研究/third_party/backbones/MFrFM` | Code inspected. |
| EEG-DINO | unresolved | False | `` | `/ai/224duibishiyan/615新研究/third_party/backbones/EEG-DINO` | Official GitHub unresolved; HF page https://huggingface.co/eegdino/EEG-DINO found. |

## Dataset Inventory

| Dataset | Task | Priority | Status | Files | Format | Path |
| --- | --- | --- | --- | ---: | --- | --- |
| BCIC-IV-2a / BNCI2014-001 | MI | primary | ready | 36 | .mat | `/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data` |
| PhysioNetMI / EEGMMI | MI | secondary | ready | 3052 | .edf;.event | `/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files` |
| CHB-MIT | seizure | optional | likely_ready | 1377 | .csv;.edf | `/ai/224duibishiyan` |
| DEAP | emotion | optional | likely_ready | 1 | .npy | `/ai/224duibishiyan/新研究/EEG/EEG_OSR_Mainline_Active/external_repos/eeg_external_experts/PESD/DATA` |
| P300 / ERP | p300 | optional | likely_ready | 632 | .csv;.edf;.mat;.npy;.npz;.pkl;.set | `/ai/224duibishiyan` |
| SEED / SEED-IV | emotion | optional | likely_ready | 7862 | .cnt;.csv;.mat;.npy;.npz;.pkl | `/ai/224duibishiyan` |
| Sleep / ISRUC | sleep | optional | likely_ready | 7570 | .edf;.npy | `/ai/224duibishiyan` |
| ... | ... | ... | ... | ... | ... | omitted 1467 low-priority/excluded scan groups; see dataset_inventory.csv/json |

## MI Dataset Order

- first: **BCIC-IV-2a / BNCI2014-001** at `/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data` status `ready` score `100`
- second: **PhysioNetMI / EEGMMI** at `/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files` status `ready` score `75`
- third: none

## Recommended Next Experiment

- backbone: `MIRepNet`
- dataset: `BCIC-IV-2a / BNCI2014-001`
- reason: Use MIRepNet first if official weight is ready; otherwise continue with CBraMod on the complete BCIC-IV-2a inventory.
- blocked_by: `[]`

## Protocol Safety

- raw_dataset_copied: `False`
- old_outputs_modified: `False`
- training_run_started: `False`

## Warnings

- EEG-DINO: unresolved

## Compact JSON

```json
{
  "backbones": {
    "CBraMod": {
      "git_commit": "",
      "has_weights": true,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/sas_cert_cbramod_mve/third_party/CBraMod",
      "status": "ready"
    },
    "EEG-DINO": {
      "git_commit": "",
      "has_weights": false,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/third_party/backbones/EEG-DINO",
      "status": "unresolved"
    },
    "EEGPT": {
      "git_commit": "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705",
      "has_weights": false,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/third_party/backbones/EEGPT",
      "status": "code_only"
    },
    "LaBraM": {
      "git_commit": "c431221e6cfd23dbfa9950e0180682fb322b0548",
      "has_weights": true,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/third_party/backbones/LaBraM",
      "status": "ready"
    },
    "MFrFM": {
      "git_commit": "aab96df836e88f19527cf311f2c54cc20fec59cd",
      "has_weights": false,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/third_party/backbones/MFrFM",
      "status": "code_only"
    },
    "MIRepNet": {
      "git_commit": "2bf16bcea493911948bfc9230775f56d47beeade",
      "has_weights": true,
      "local_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/third_party/backbones/MIRepNet",
      "status": "ready"
    }
  },
  "datasets": {
    "non_mi_candidates": [
      "CHB-MIT",
      "DEAP",
      "P300 / ERP",
      "SEED / SEED-IV",
      "Sleep / ISRUC"
    ],
    "primary_mi_dataset": "BCIC-IV-2a / BNCI2014-001",
    "primary_mi_path": "/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data",
    "secondary_mi_dataset": "PhysioNetMI / EEGMMI",
    "secondary_mi_path": "/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files",
    "third_mi_dataset": "",
    "third_mi_path": ""
  },
  "project_root": "/ai/224duibishiyan/615\u65b0\u7814\u7a76",
  "protocol_safety": {
    "old_outputs_modified": false,
    "raw_dataset_copied": false,
    "training_run_started": false
  },
  "recommended_next_experiment": {
    "backbone": "MIRepNet",
    "blocked_by": [],
    "dataset": "BCIC-IV-2a / BNCI2014-001",
    "reason": "Use MIRepNet first if official weight is ready; otherwise continue with CBraMod on the complete BCIC-IV-2a inventory."
  },
  "stage": "step0_backbone_dataset_inventory",
  "status": "completed",
  "warnings": [
    "EEG-DINO: unresolved"
  ],
  "workspace_root": "/ai/224duibishiyan/615\u65b0\u7814\u7a76"
}
```
