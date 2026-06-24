# MIRepNet Adapter Audit

- adapter_ready: `False`
- reason: not_generated_until_validated_inverse_distance_interpolation_and_EA_are_implemented
- missing_dependencies: `['wandb', 'einops']`
- raw BCIC path: `/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014`
- input_raw_shape: `{'status': 'audited', 'files': 18, 'raw_shape_examples': [[96735, 25], [96735, 25], [96735, 25], [96735, 25], [96735, 25]], 'valid_trials': 5088, 'left_right_trials': 2544, 'labels': [0, 1, 2, 3]}`
- expected processed shape: `{'BNCI2014001_2class_left_right': '[samples,45,1000]', 'BNCI2014001_4class_full': '[samples,45,1000]'}`
- selected tasks: `['BNCI2014001_2class_left_right', 'BNCI2014001_4class_full']`
- directly observed template channels: `['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2']`
- interpolated channels required: `['F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC6', 'FT8', 'T7', 'T8', 'TP7', 'CP5', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P4', 'P6', 'P8']`
- EA behavior: `PAPER_PROTOCOL_EA requested; official behavior for train/test separation must be verified before generating exact arrays`
- generated files: `[]`

No X.npy/labels.npy were generated in this gate because doing so without a validated inverse-distance interpolation and exact EA split behavior would create a hybrid artifact that could be mistaken for paper_exact.
