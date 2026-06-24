# Paper Code Run Report

## Scope

This pass ran the official CBraMod `finetune_main.py` entrypoint on the existing processed LMDB datasets:

- BCIC-IV-2a: `/ai/224duibishiyan/CBraMod-main/data/processed_bciciv2a_40`
- PhysioNet-MI: `/ai/224duibishiyan/CBraMod-main/data/PhysioNet_MI/processed_average`

Protocol label: `code_exact_on_existing_paper_aligned_lmdb`.

This is not labeled as strict `paper_exact`, because the LMDB files were existing local processed artifacts rather than regenerated from raw data during this run. It is still a useful code-first reproduction because the official code path, seed, default classifier, optimizer, LR, weight decay, label smoothing, and 50-epoch schedule were used.

## Results

| Model | Dataset | Test balanced acc | Test kappa | Test weighted F1 | Paper acc | Paper kappa | Paper F1 | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CBraMod | BCIC-IV-2a | 0.39149 | 0.18866 | 0.33078 | 0.5138 | 0.3518 | 0.4984 | Large gap |
| CBraMod | PhysioNet-MI | 0.62847 | 0.50455 | 0.62847 | 0.6417 | 0.5222 | 0.6427 | Close |

The CBraMod evaluator reports `acc` as `balanced_accuracy_score` and `f1` as weighted F1, so the metric names align with the paper target table.

## Interpretation

PhysioNet-MI reproduced reasonably well. The run is slightly below the paper on all three metrics, but the gaps are small: balanced accuracy -1.32pp, kappa -0.0177, weighted F1 -1.42pp. All three are within 2x the paper-reported standard deviation.

BCIC-IV-2a did not reproduce the paper E.9 result. The final test balanced accuracy was 39.15%, versus paper 51.38%. Kappa and weighted F1 were also far below the paper target. Validation peaked at epoch 5 with balanced accuracy 54.43%, then the best checkpoint produced only 39.15% on the test split. That points to a strong validation-test split/generalization issue for A08-A09, not a simple training failure.

## Important Caveats

- This pass did not run MIRepNet. The MIRepNet paper-code run still needs a full 45-channel EA adapter dataset packaged as the official `X.npy` and `labels.npy` inputs.
- This pass did not regenerate paper-exact LMDBs from raw data. For strict paper-exact claims, regenerate/audit raw preprocessing first.
- The BCIC-IV-2a LOSO key-ingredients protocol is separate from the E.9 split and requires EA, session statistics, Mixup, and subject-wise regularization; it was not run here.

## Output Files

- Metrics CSV: `outputs/paper_code_runs/paper_code_run_metrics.csv`
- Compact JSON: `outputs/paper_code_runs/compact_paper_code_run_result.json`
- BCIC log: `outputs/paper_code_runs/cbramod_bcic_iv_2a_e9_seed3407/stdout.log`
- PhysioNet log: `outputs/paper_code_runs/cbramod_physionet_mi_seed3407/stdout.log`
