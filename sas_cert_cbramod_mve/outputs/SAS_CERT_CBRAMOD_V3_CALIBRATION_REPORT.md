# SAS-Cert-CBraMod V3 Calibration Repair Report

## Summary

- Status: `completed`
- Score variant: `artifact_gate_content_rank`
- Main method: `SoftWeightArtifactReject`
- Best calibration group: `SWAR_LabelSmoothing_eps010`
- Decision: `GO_CALIBRATED_SOFTWEIGHT`
- Protocol leakage detected: `False`

## Why V2 Was Not Confirmatory

- V2 SoftWeightArtifactReject vs NaiveAug Acc delta was `+0.0179`.
- V2 Macro-F1 delta was `+0.0204`.
- V2 ECE delta was `+0.0146`, above the +0.01 gate.

## V3 Group Decisions

| Group | Acc delta | Macro-F1 delta | ECE delta | NLL delta | Brier delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ArtifactReject_raw | +0.0095 | +0.0252 | +0.0003 | -0.0009 | -0.0004 | ARTIFACT_ONLY_BRANCH_DIAGNOSTIC |
| SWAR_BrierMix_lam005 | +0.0077 | +0.0132 | +0.0013 | -0.0011 | -0.0005 | GO_CALIBRATED_SOFTWEIGHT |
| SWAR_BrierMix_lam010 | +0.0066 | +0.0138 | +0.0015 | -0.0009 | -0.0004 | GO_CALIBRATED_SOFTWEIGHT |
| SWAR_LabelSmoothing_eps005 | +0.0050 | +0.0138 | -0.0009 | -0.0010 | -0.0005 | GO_CALIBRATED_SOFTWEIGHT |
| SWAR_LabelSmoothing_eps010 | +0.0098 | +0.0183 | +0.0020 | -0.0013 | -0.0006 | GO_CALIBRATED_SOFTWEIGHT |
| SWAR_TempScale_SourceVal | +0.0049 | +0.0072 | +0.0046 | +0.0108 | +0.0056 | CALIBRATION_OVERREGULARIZED |
| SoftWeightArtifactReject_raw | +0.0046 | +0.0088 | -0.0049 | -0.0013 | -0.0006 | CALIBRATION_OVERREGULARIZED |

## Best Group

- Best calibration group: `SWAR_LabelSmoothing_eps010`
- Delta Acc: `+0.0098`
- Delta Macro-F1: `+0.0183`
- Delta ECE: `+0.0020`
- Delta NLL: `-0.0013`
- Delta Brier: `-0.0006`

## Temperature Scaling

- Temperature mean: `2.662962962962963`
- Temperature std: `1.2586436403696928`
- Temperature was fitted only on source validation, never on target E/test.

## Artifact Branch

- ArtifactReject_raw remains diagnostic: Acc delta `+0.0095`, Macro-F1 delta `+0.0252`, ECE delta `+0.0003`, decision `ARTIFACT_ONLY_BRANCH_DIAGNOSTIC`.
- Raw SWAR before calibration: Acc delta `+0.0046`, Macro-F1 delta `+0.0088`, ECE delta `-0.0049`.

## Physio Note

- Physio original AUC: `0.6346`
- Physio fixed BadPhysio AUC: `0.0757`
- REFINE_PHYSIO remains recommended because the fixed/variant diagnostics still show weak BadPhysio behavior.

## Next Action

run longer epochs or more seeds with the winning calibrated SWAR group; continue REFINE_PHYSIO by splitting bandpower/covariance/topology/channel_order

## Compact JSON

```json
{
  "artifact_reject_percentile": 10.0,
  "backbone": "CBraMod",
  "backbone_frozen": true,
  "baseline": "NaiveAug",
  "best_calibration_group": "SWAR_LabelSmoothing_eps010",
  "best_group_delta_acc": 0.00977366255144041,
  "best_group_delta_brier": -0.0006306710915843228,
  "best_group_delta_ece": 0.0019956481156280376,
  "best_group_delta_macro_f1": 0.018269399839661,
  "best_group_delta_nll": -0.0012782193996287994,
  "dataset": "BCIC-IV-2a",
  "decision": "GO_CALIBRATED_SOFTWEIGHT",
  "main_method": "SoftWeightArtifactReject",
  "next_action": "run longer epochs or more seeds with the winning calibrated SWAR group; continue REFINE_PHYSIO by splitting bandpower/covariance/topology/channel_order",
  "project": "sas_cert_cbramod_mve",
  "protocol_leakage_detected": false,
  "score_variant": "artifact_gate_content_rank",
  "stage": "v3_calibration_repair",
  "status": "completed",
  "temperature_mean": 2.662962962962963,
  "temperature_std": 1.2586436403696928,
  "w_min": 0.2
}
```
