# SAS-Cert-CBraMod V2 SoftWeight ArtifactReject Report

## Summary

- Best score variant: `artifact_gate_content_rank`
- Physio original AUC: `0.6346`
- Physio fixed used: `False`
- Final decision: `REFINE_PHYSIO, STOP_HARD_TOP50, ARTIFACT_ONLY_BRANCH`
- Protocol leakage detected: `False`

## Why Hard Top50 Failed

- Layer 2 original overall AUC was `0.6217`, below the 0.70 gate.
- SASCertTop50 vs NaiveAug Acc was `-0.0237`.
- SASCertTop50 vs Random50 Acc was `+0.0091`, so hard Top50 only beat random selection weakly and did not beat using all augmentations.

## Layer 2-v2 Diagnosis

| Variant | Overall clean-vs-bad AUC |
| --- | ---: |
| artifact_gate_content_rank | 0.7239 |
| current_total | 0.6217 |
| artifact_content_physio_fixed | 0.5903 |
| artifact_content_style | 0.5750 |
| artifact_content | 0.5607 |

- Physio direction maybe wrong: `False`
- Physio fixed adopted: `False`
- Physio fixed BadPhysio AUC: `0.0757`

## Layer 3-v2 Training

| Comparison | Acc delta | Macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: |
| SoftWeightArtifactReject - NaiveAug | +0.0179 | +0.0204 | +0.0146 |
| ArtifactReject - NaiveAug | +0.0157 | +0.0304 | +0.0149 |

## ArtifactReject Calibration Risk

- ArtifactReject calibration risk: `True`
- It is not the main method when ECE delta exceeds `+0.01`; observed ECE delta is `+0.0149`.

## Decision Rules

- GO_SOFTWEIGHT: `False`
- REFINE_PHYSIO: `True`
- STOP_HARD_TOP50: `True`
- ARTIFACT_ONLY_BRANCH: `True`

## Next Action

split BadPhysio into bandpower/covariance/topology/channel_order diagnostics; test calibration-aware loss or temperature scaling for artifact-focused branch; stop treating hard SASCertTop50 as main method

## Compact JSON

```json
{
  "artifactreject_minus_naive": {
    "acc": 0.015689300411522635,
    "calibration_risk": true,
    "ece": 0.014927714920330626,
    "macro_f1": 0.030430012862597316
  },
  "best_score_variant": "artifact_gate_content_rank",
  "final_decision": [
    "REFINE_PHYSIO",
    "STOP_HARD_TOP50",
    "ARTIFACT_ONLY_BRANCH"
  ],
  "hard_top50_minus_naive_acc": -0.02366255144032922,
  "layer2_v2_overall_auc_by_variant": {
    "artifact_content": 0.5607356824417009,
    "artifact_content_physio_fixed": 0.5902969821673525,
    "artifact_content_style": 0.5750219478737997,
    "artifact_gate_content_rank": 0.7238972908093279,
    "current_total": 0.6216899862825789
  },
  "next_action": "split BadPhysio into bandpower/covariance/topology/channel_order diagnostics; test calibration-aware loss or temperature scaling for artifact-focused branch; stop treating hard SASCertTop50 as main method",
  "physio_fixed_badphysio_auc": 0.075730109739369,
  "physio_fixed_used": false,
  "physio_original_auc": 0.6345891632373114,
  "project": "sas_cert_cbramod_mve",
  "protocol_leakage_detected": false,
  "softweight_artifactreject_minus_naive": {
    "acc": 0.01787551440329218,
    "ece": 0.014589487222395075,
    "macro_f1": 0.02043688546983737
  },
  "softweight_artifactreject_minus_random50_acc": 0.037808641975308636,
  "softweight_artifactreject_minus_sascerttop50_acc": 0.020190329218106987,
  "status": "completed",
  "version": "v2_softweight_artifact_reject"
}
```
