# SAS-Cert-CBraMod V4 Confirmatory MVE And Physio Forensics

## Summary

- Status: `completed`
- Main method: `SASCert_SoftAR_LS010`
- Decision: `GO_STABLE_CALIBRATED_SOFTWEIGHT`
- Protocol leakage detected: `False`

## Why V3 Was GO

- V3 best group was `SWAR_LabelSmoothing_eps010`.
- V3 Macro-F1 delta was `+0.0183` with ECE delta `+0.0020`.
- V3 NLL/Brier deltas were `-0.0013` / `-0.0006`.

## V4 Confirmatory Result

| Metric | Delta vs NaiveAug |
| --- | ---: |
| Acc | +0.0137 |
| Macro-F1 | +0.0237 |
| ECE | +0.0045 |
| NLL | -0.0009 |
| Brier | -0.0004 |
| Worst-subject Acc | +0.0083 |
| Subject-wise Std | +0.0022 |

- Subject win rate Acc: `1.0000`
- Subject win rate Macro-F1: `1.0000`

## Artifact Diagnostic Branch

- ArtifactReject_raw_diagnostic Acc delta: `+0.0150`
- ArtifactReject_raw_diagnostic Macro-F1 delta: `+0.0324`
- ArtifactReject_raw_diagnostic ECE delta: `+0.0044`
- Decision: `ARTIFACT_ONLY_BRANCH_DIAGNOSTIC`

## Physio Forensics

| Bad type | Subscore | AUC | Inverted AUC | Direction issue | Usable |
| --- | --- | ---: | ---: | --- | --- |
| BadPhysio_bandpower | physio_bandpower | 0.6379 | 0.3621 | False | False |
| BadPhysio_covariance | physio_covariance | 0.9999 | 0.0001 | False | True |
| BadPhysio_topology | physio_topology | 0.6931 | 0.3069 | False | False |
| BadPhysio_channel_order | physio_channel_order | 0.1149 | 0.8851 | True | False |

- Physio best component: `physio_covariance` AUC `0.9999`
- Physio ready for training: `True`

## Final Decision

`GO_STABLE_CALIBRATED_SOFTWEIGHT`

## Next Action

enter longer epoch or more seeds confirmatory run with SASCert_SoftAR_LS010

## Compact JSON

```json
{
  "artifact_reject_percentile": 10,
  "backbone": "CBraMod",
  "backbone_frozen": true,
  "baseline": "NaiveAug",
  "dataset": "BCIC-IV-2a",
  "decision": "GO_STABLE_CALIBRATED_SOFTWEIGHT",
  "delta_acc": 0.013657407407407396,
  "delta_brier": -0.00038867577089130023,
  "delta_ece": 0.004534982561053322,
  "delta_macro_f1": 0.023688857809868236,
  "delta_nll": -0.0009087483088177084,
  "delta_subject_wise_std": 0.002171519514135911,
  "delta_worst_subject_acc": 0.008333333333333331,
  "label_smoothing": 0.1,
  "main_method": "SASCert_SoftAR_LS010",
  "next_action": "enter longer epoch or more seeds confirmatory run with SASCert_SoftAR_LS010",
  "physio_best_component": "physio_covariance",
  "physio_best_component_auc": 0.999920987654321,
  "physio_ready_for_training": true,
  "project": "sas_cert_cbramod_mve",
  "protocol_leakage_detected": false,
  "score_variant": "artifact_gate_content_rank",
  "stage": "v4_confirmatory_mve",
  "status": "completed",
  "subject_win_rate_acc": 1.0,
  "subject_win_rate_macro_f1": 1.0,
  "w_min": 0.2
}
```
