# Step1 Backbone Dataset Compatibility

## Compatibility Matrix

| Backbone | Dataset | Status | Forward | Feature | Missing deps | Adapter flags | Recommended | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBraMod | BCIC-IV-2a | ready | True | `[1, 200]` |  | - | True | Frozen CBraMod forward smoke succeeded on one BCIC-IV-2a trial. |
| MIRepNet | BCIC-IV-2a | needs_adapter | False | `` | wandb;einops | channel_mapping, resampling, crop | False | MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule. |
| CBraMod | PhysioNetMI | ready | True | `[1, 200]` |  | channel_mapping, resampling, crop | False | Forward succeeded after deterministic 22-channel BNCI-style mapping and crop/resample. mapped_channels=['Fz..', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp3.', 'Cp1 |
| MIRepNet | PhysioNetMI | needs_adapter | False | `` | wandb;einops | channel_mapping, resampling, crop | False | MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule. |
| LaBraM | BCIC-IV-2a | code_ready_weight_ready | False | `` |  | channel_mapping | False | Import/checkpoint-path audit only. LaBraM requires channel-order adapter/input_chans handling; no forward attempted per Step1. |

## BCIC-IV-2a Loader

- status: `ready`
- path: `/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014`
- subjects: `['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']`
- sessions: `['E', 'T']`
- trial_shape: `[22, 800]`
- labels: `[0, 1, 2, 3]`
- EOG excluded: `True`

## PhysioNetMI Loader

- status: `ready`
- path: `/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files`
- subjects: `109`
- runs_per_subject: `[14]`
- mi_runs_detected: `[4, 6, 8, 10, 12, 14]`
- recommended_task: 2-class left vs right from MI runs 4/8/12 first; then 2-class hands vs feet from runs 6/10/14; 4-class requires combining run semantics carefully.
- recommended_window: 4.0s trial window, resample/crop to 200Hz x 4s = 800 samples for CBraMod compatibility

## Recommended Next

- backbone: `CBraMod`
- dataset: `BCIC-IV-2a / BNCI2014-001`
- reason: CBraMod x BCIC-IV-2a is ready while MIRepNet needs adapter/failed; continue CBraMod + BCIC-IV-2a v5 locked confirmatory.

## Protocol Safety

- raw_dataset_copied: `False`
- training_started: `False`
- old_outputs_modified: `False`

## Warnings

- MIRepNet x BCIC-IV-2a missing_dependencies=wandb;einops
- MIRepNet x BCIC-IV-2a status=needs_adapter: MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule.
- MIRepNet x PhysioNetMI missing_dependencies=wandb;einops
- MIRepNet x PhysioNetMI status=needs_adapter: MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule.
- LaBraM x BCIC-IV-2a status=code_ready_weight_ready: Import/checkpoint-path audit only. LaBraM requires channel-order adapter/input_chans handling; no forward attempted per Step1.

## Compact JSON

```json
{
  "bcic2a": {
    "labels": [
      0,
      1,
      2,
      3
    ],
    "path": "/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014",
    "ready": true,
    "sessions": [
      "E",
      "T"
    ],
    "status": "ready",
    "subjects": 9,
    "trial_shape": [
      22,
      800
    ]
  },
  "failed_pairs": [],
  "needs_adapter_pairs": [
    "MIRepNet x BCIC-IV-2a",
    "MIRepNet x PhysioNetMI",
    "LaBraM x BCIC-IV-2a"
  ],
  "physionetmi": {
    "mi_runs_detected": [
      4,
      6,
      8,
      10,
      12,
      14
    ],
    "path": "/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files",
    "ready": true,
    "recommended_task": "2-class left vs right from MI runs 4/8/12 first; then 2-class hands vs feet from runs 6/10/14; 4-class requires combining run semantics carefully.",
    "runs_per_subject": [
      14
    ],
    "status": "ready",
    "subjects": 109
  },
  "protocol_safety": {
    "old_outputs_modified": false,
    "raw_dataset_copied": false,
    "training_started": false
  },
  "ready_pairs": [
    "CBraMod x BCIC-IV-2a",
    "CBraMod x PhysioNetMI"
  ],
  "recommended_next": {
    "backbone": "CBraMod",
    "dataset": "BCIC-IV-2a / BNCI2014-001",
    "reason": "CBraMod x BCIC-IV-2a is ready while MIRepNet needs adapter/failed; continue CBraMod + BCIC-IV-2a v5 locked confirmatory."
  },
  "stage": "step1_backbone_dataset_smoke",
  "status": "completed",
  "tested_pairs": [
    "CBraMod x BCIC-IV-2a",
    "MIRepNet x BCIC-IV-2a",
    "CBraMod x PhysioNetMI",
    "MIRepNet x PhysioNetMI",
    "LaBraM x BCIC-IV-2a"
  ],
  "warnings": [
    "MIRepNet x BCIC-IV-2a missing_dependencies=wandb;einops",
    "MIRepNet x BCIC-IV-2a status=needs_adapter: MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule.",
    "MIRepNet x PhysioNetMI missing_dependencies=wandb;einops",
    "MIRepNet x PhysioNetMI status=needs_adapter: MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule.",
    "LaBraM x BCIC-IV-2a status=code_ready_weight_ready: Import/checkpoint-path audit only. LaBraM requires channel-order adapter/input_chans handling; no forward attempted per Step1."
  ],
  "workspace_root": "/ai/224duibishiyan/615新研究"
}
```
