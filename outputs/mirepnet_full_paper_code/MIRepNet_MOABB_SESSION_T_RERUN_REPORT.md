# MIRepNet BNCI2014001-4 MOABB Session-T Rerun Report

## Protocol
- Adapter: MOABB `BNCI2014_001`, `MotorImagery(n_classes=4, fmin=8, fmax=30)`.
- Session: only `0train`, matching the author issue guidance for BNCI2014001 / BNCI2014001-4.
- Saved input: `[2592, 22, 1000]`, labels balanced 648 per class, 288 trials per subject.
- Official MIRepNet flow kept: subject-wise `30% train / 70% test`, then EA and 45-channel interpolation separately for train/test.
- Runs: batch8 default-like and batch32 author-suggested range; both use seed 666, epochs 10, lr 0.001, cosine scheduler.

## Result Summary
| Run | Mean Acc | vs previous raw/T+E hybrid | vs paper 64.14 |
| --- | ---: | ---: | ---: |
| Previous raw/T+E hybrid | 58.38% | +0.00 pp | -5.76 pp |
| MOABB session-T batch32 | 57.76% | -0.62 pp | -6.38 pp |
| MOABB session-T batch8 | 60.23% | +1.85 pp | -3.91 pp |

## Batch8 Subject Accuracy
- Subject_0_Acc: 67.33%
- Subject_1_Acc: 48.02%
- Subject_2_Acc: 78.71%
- Subject_3_Acc: 48.51%
- Subject_4_Acc: 44.55%
- Subject_5_Acc: 43.56%
- Subject_6_Acc: 82.18%
- Subject_7_Acc: 76.24%
- Subject_8_Acc: 52.97%

## Batch32 Subject Accuracy
- Subject_0_Acc: 67.82%
- Subject_1_Acc: 48.51%
- Subject_2_Acc: 78.71%
- Subject_3_Acc: 41.58%
- Subject_4_Acc: 47.03%
- Subject_5_Acc: 41.09%
- Subject_6_Acc: 76.73%
- Subject_7_Acc: 64.85%
- Subject_8_Acc: 53.47%

## Interpretation
The main protocol issue has now been corrected: the input is MOABB first/session-T only, with exactly 2592 trials, and MIRepNet still does EA and 45-channel interpolation after the 30/70 split. However, the corrected reruns did not improve the mean accuracy over the previous raw T+E hybrid run. The default-like batch8 run is lower than both batch32 and the previous hybrid. So the earlier performance gap is not explained by session selection alone. The remaining likely causes are differences between the authors' unreleased processed dataset package and this reconstructed MOABB adapter, split/seed sensitivity, and possibly subtle label/channel/time-window preprocessing choices.

## Outputs
- Batch8 metrics CSV: `/ai/224duibishiyan/615新研究/outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_4class_moabb_session_t_gpu_batch8_metrics.csv`
- Batch32 metrics CSV: `/ai/224duibishiyan/615新研究/outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_4class_moabb_session_t_gpu_batch32_metrics.csv`
- Compact JSON: `/ai/224duibishiyan/615新研究/outputs/mirepnet_full_paper_code/compact_mirepnet_moabb_session_t_rerun_result.json`
- Adapter manifest: `/ai/224duibishiyan/615新研究/outputs/mirepnet_full_paper_code/mirepnet_bnci2014001_moabb_session_t_adapter_manifest.json`
