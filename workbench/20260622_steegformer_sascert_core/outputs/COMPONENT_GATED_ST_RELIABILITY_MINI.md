# Component-Gated ST Reliability Mini

## Scope

- Dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`
- Backbone: frozen source-tuned `ST-EEGFormer-small`
- Targets: `90,91,92`
- Seeds: `20,21`
- Support: `5-shot` per class
- Candidate pool: same support split and same augmentation candidates for all groups
- Protocol: target held-out trials were used only for final evaluation

## Groups

| Group | Rule |
|---|---|
| `NaiveAug_LS010` | support plus all augmentation candidates, unit weights |
| `SoftWeight_noReject_LS010` | all candidates weighted by content rank, no rejection |
| `SASCert_SoftAR_LS010` | artifact p90 reject, remaining candidates weighted by content rank |
| `ArtifactGatePhysio_LS010` | physio score with artifact p90 candidates set to zero, then soft weighting |
| `ComponentGatedV1_LS010` | `ranknorm(0.75 * physio + 0.25 * style) * artifact_gate_pass`, then soft weighting |

## Result

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7869 | 0.7862 | 0.1413 | 0.5065 | 0.3135 |
| `SoftWeight_noReject_LS010` | 0.7902 | 0.7898 | 0.1471 | 0.4908 | 0.2959 |
| `SASCert_SoftAR_LS010` | 0.7860 | 0.7852 | 0.1459 | 0.4966 | 0.3012 |
| `ArtifactGatePhysio_LS010` | 0.7823 | 0.7817 | 0.1385 | 0.5048 | 0.3151 |
| `ComponentGatedV1_LS010` | 0.7823 | 0.7817 | 0.1473 | 0.5093 | 0.3178 |

## Key Comparisons

| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Macro-F1 Win Rate |
|---|---:|---:|---:|---:|---:|
| `ComponentGatedV1 - NaiveAug` | -0.0046 | -0.0046 | +0.0060 | +0.0027 | 0.1667 |
| `ComponentGatedV1 - SoftWeight_noReject` | -0.0080 | -0.0081 | +0.0002 | +0.0185 | 0.0000 |
| `ComponentGatedV1 - SASCert_SoftAR` | -0.0038 | -0.0036 | +0.0014 | +0.0127 | 0.1667 |
| `ComponentGatedV1 - ArtifactGatePhysio` | 0.0000 | 0.0000 | +0.0088 | +0.0045 | 0.0000 |

## Decision

`do_not_expand_component_gated_or_artifact_gate_physio_on_st`

The score-only audit said `component_gated_v1` and `artifact_gate_physio` are much better clean-vs-bad detectors than the old scalar score. The training mini says that this does not automatically become a better augmentation weighting rule for ST-EEGFormer-small. In this branch, the simplest no-reject soft weighting still has the best mean BAcc, Macro-F1, NLL, and Brier.

This is a useful failure: the certificate can be good at detecting synthetic bad samples while still being too harsh or misaligned for training utility. The next ST branch should avoid adding more gates and should instead confirm the current `SoftWeight_noReject_LS010` behavior with cleaner reporting.
