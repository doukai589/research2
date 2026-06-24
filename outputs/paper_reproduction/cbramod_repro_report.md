# CBraMod Paper Reproduction Report

## Protocol Separation

### code_exact
- PhysioNet-MI: `available_but_requires_processed_lmdb`; entrypoint `/ai/224duibishiyan/615新研究/third_party/CBraMod-main/finetune_main.py`; Official entry defaults are code behavior; requires LMDB datasets_dir with train/val/test keys.
- BCIC-IV-2a: `available_but_requires_processed_lmdb`; entrypoint `/ai/224duibishiyan/615新研究/third_party/CBraMod-main/finetune_main.py`; Official code uses LMDB dataset class; default CLI is not BCIC.

### paper_exact
- PhysioNet-MI: `needs_paper_preprocessing_lmdb`; entrypoint `paper raw EDF preprocessing + CBraMod full finetune`; Raw EDF split 1-70/71-89/90-109 must be converted to exact LMDB before exact run.
- BCIC-IV-2a: `needs_paper_preprocessing_lmdb`; entrypoint `paper raw MAT preprocessing + CBraMod full finetune`; Need exact [2,6]s, 0.3-40Hz, 200Hz, 1-5/6-7/8-9 split before exact run.

### executable_hybrid
- PhysioNet-MI: `forward_only_not_reproduction`; entrypoint `Step1 mapped/cropped forward smoke`; Can forward with 22-channel mapping, but this is not paper exact because paper uses 64 channels.
- BCIC-IV-2a: `forward_only_not_reproduction`; entrypoint `Step1 raw MAT loader + CBraMod forward smoke`; Forward works, but not full fine-tuning reproduction.

## Results

| Dataset | Protocol | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PhysioNet-MI | code_exact | official_code_default | not_run |  |  | not_run | not_run_compute_guard |
| PhysioNet-MI | paper_exact | 4-class | balanced_accuracy | 0.6417±0.0091 |  | failed | needs_paper_preprocessing_lmdb |
| PhysioNet-MI | paper_exact | 4-class | cohen_kappa | 0.5222±0.0169 |  | failed | needs_paper_preprocessing_lmdb |
| PhysioNet-MI | paper_exact | 4-class | weighted_f1 | 0.6427±0.01 |  | failed | needs_paper_preprocessing_lmdb |
| PhysioNet-MI | executable_hybrid | forward_smoke_only | not_a_metric |  |  | not_scored | hybrid_forward_ready |
| BCIC-IV-2a | code_exact | official_code_default | not_run |  |  | not_run | not_run_compute_guard |
| BCIC-IV-2a | paper_exact | E.9 4-class | balanced_accuracy | 0.5138±0.0066 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | paper_exact | E.9 4-class | cohen_kappa | 0.3518±0.0094 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | paper_exact | E.9 4-class | weighted_f1 | 0.4984±0.0085 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | executable_hybrid | forward_smoke_only | not_a_metric |  |  | not_scored | hybrid_forward_ready |
| BCIC-IV-2a LOSO key ingredients | paper_exact | LOSO key ingredients | balanced_accuracy | 0.7405±0.0635 |  | failed | not_run_missing_key_ingredients_implementation |
| BCIC-IV-2a LOSO key ingredients | paper_exact | LOSO key ingredients | auc_pr | 0.5997±0.0833 |  | failed | not_run_missing_key_ingredients_implementation |
| BCIC-IV-2a LOSO key ingredients | paper_exact | LOSO key ingredients | auroc | 0.7195±0.0682 |  | failed | not_run_missing_key_ingredients_implementation |
