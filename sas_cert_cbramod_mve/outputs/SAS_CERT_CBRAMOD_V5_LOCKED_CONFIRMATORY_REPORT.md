# SAS-Cert-CBraMod V5 Locked Confirmatory Report

## Summary

- Status: `completed`
- Main method: `SASCert_SoftAR_LS010`
- Decision: `LABEL_SMOOTHING_CONFOUNDED`
- Protocol leakage detected: `False`

## Why V4 Was GO

- V4 Acc/Macro-F1 deltas were `+0.0137` / `+0.0237`.
- V4 ECE/NLL/Brier deltas were `+0.0045` / `-0.0009` / `-0.0004`.
- V4 subject win rates were `1.0000` / `1.0000`.

## Locked Method Vs Baselines

| Baseline | Acc | Macro-F1 | ECE | NLL | Brier | Subject Win Acc/F1 | Seed Win Acc/F1 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NaiveAug_raw | +0.0111 | +0.0211 | +0.0084 | -0.0011 | -0.0005 | 0.8889/0.8889 | 0.6667/0.7333 | GO_RAW_BASELINE |
| NaiveAug_LS010 | +0.0072 | +0.0192 | +0.0068 | -0.0009 | -0.0005 | 0.6667/0.8889 | 0.7333/0.8667 | CALIBRATION_REGRESSION |

## Label Smoothing Confound Check

- Raw baseline decision: `GO_RAW_BASELINE`
- LS baseline decision: `CALIBRATION_REGRESSION`
- If the LS baseline fails while raw passes, the report must treat the result as label-smoothing-confounded.

## Artifact Diagnostic Branch

- ArtifactReject_raw_diagnostic vs NaiveAug_raw Acc/Macro-F1: `+0.0121` / `+0.0294`.
- ECE/NLL/Brier: `+0.0059` / `-0.0013` / `-0.0006`.
- Decision: `ARTIFACT_ONLY_BRANCH_DIAGNOSTIC`.

## Physio Covariance Audit

- physio_covariance_auc: `1.0000`
- covariance_detector_is_construct_specific: `False`
- physio_ready_for_training: `True`
- conclusion: `cross_physio_signal_present`

## Final Decision

`LABEL_SMOOTHING_CONFOUNDED`

## Next Action

keep NaiveAug_LS010 as all future main baseline

## Compact JSON

```json
{
  "artifact_reject_percentile": 10,
  "backbone": "CBraMod",
  "backbone_frozen": true,
  "baseline_ls": "NaiveAug_LS010",
  "baseline_raw": "NaiveAug_raw",
  "covariance_detector_is_construct_specific": false,
  "dataset": "BCIC-IV-2a",
  "decision": "LABEL_SMOOTHING_CONFOUNDED",
  "delta_acc_vs_ls": 0.0071759259259259744,
  "delta_acc_vs_raw": 0.011059670781893016,
  "delta_brier_vs_ls": -0.000454219756595875,
  "delta_brier_vs_raw": -0.00048493048799325145,
  "delta_ece_vs_ls": 0.0067911030211075946,
  "delta_ece_vs_raw": 0.008431449058032586,
  "delta_macro_f1_vs_ls": 0.019224317595990942,
  "delta_macro_f1_vs_raw": 0.021066332534534077,
  "delta_nll_vs_ls": -0.0008628668608490209,
  "delta_nll_vs_raw": -0.0011058180420486785,
  "label_smoothing": 0.1,
  "main_method": "SASCert_SoftAR_LS010",
  "next_action": "keep NaiveAug_LS010 as all future main baseline",
  "physio_covariance_auc": 0.999966049382716,
  "physio_ready_for_training": true,
  "project": "sas_cert_cbramod_mve",
  "protocol_leakage_detected": false,
  "score_variant": "artifact_gate_content_rank",
  "seed_win_rate_acc_vs_raw": 0.6666666666666666,
  "seed_win_rate_macro_f1_vs_raw": 0.7333333333333333,
  "stage": "v5_locked_confirmatory",
  "status": "completed",
  "subject_win_rate_acc_vs_raw": 0.8888888888888888,
  "subject_win_rate_macro_f1_vs_raw": 0.8888888888888888,
  "w_min": 0.2
}
```
