# component_gated_cert_rule

## Intent

Define a component-gated SAS-Cert rule from the cross-backbone direction audit.

This is a rule-definition and score-only validation step. It does not train a
model or generate new augmented samples.

## Rule v1

```text
artifact_gate_pass = artifact_risk < fold_p90
base = 0.75 * physio_score + 0.25 * style_score
component_gated_v1 = ranknorm(base) * artifact_gate_pass
component_gated_v1_soft = ranknorm(base) * (0.2 + 0.8 * artifact_safe_rank)
content_score = diagnostic warning only, not a positive term
```

Reasoning:

- `physio_score` and `style_score` were clean-high on both locked backbones.
- `artifact_safe_score` is reliable for BadArtifact but is not a universal
  positive score for all bad types.
- `content_score` is backbone/bad-type dependent, especially for BadContent.

## Command

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_component_gated_cert_rule/01_define_and_validate_component_gated_rule.py
```

## Score-Only Results

| Backbone | Current SAS | ArtifactGatePhysio | ComponentGated v1 | ComponentGated v1 soft |
|---|---:|---:|---:|---:|
| `CBraMod_frozen` | `0.1969` | `0.9022` | `0.8395` | `0.7383` |
| `ST-EEGFormer-small_source_tuned` | `0.1662` | `0.9022` | `0.8395` | `0.7383` |

## Gate Stats

| Backbone | Aug type | Artifact reject rate | Content warning rate |
|---|---|---:|---:|
| `CBraMod_frozen` | `bad_artifact` | `0.5000` | `0.5333` |
| `CBraMod_frozen` | `bad_content` | `0.0000` | `0.8667` |
| `CBraMod_frozen` | `bad_physio` | `0.0000` | `0.7667` |
| `CBraMod_frozen` | `clean` | `0.0000` | `0.2167` |
| `ST-EEGFormer-small_source_tuned` | `bad_artifact` | `0.5000` | `0.9333` |
| `ST-EEGFormer-small_source_tuned` | `bad_content` | `0.0000` | `0.3500` |
| `ST-EEGFormer-small_source_tuned` | `bad_physio` | `0.0000` | `0.6167` |
| `ST-EEGFormer-small_source_tuned` | `clean` | `0.0000` | `0.3500` |

## Decision

`component_gated_v1_defined_score_validated`

The interpretable rule fixes the scalar-score direction failure, but it is not
the strongest score-only variant. `score_artifact_gate_physio` remains the
strongest score-only baseline. The next training step should be a small ST
reliability test comparing:

- current ST `SoftWeight_noReject_LS010`
- `score_artifact_gate_physio`
- `component_gated_v1`

Do not run a broad full expansion until this small reliability test is done.

## Primary Outputs

- `outputs/COMPONENT_GATED_CERT_RULE_V1.md`
- `outputs/compact_component_gated_cert_rule_v1.json`
- `outputs/component_gated_v1_component_auc.csv`
- `outputs/component_gated_v1_bad_type_auc.csv`
- `outputs/component_gated_v1_gate_stats.csv`
- `outputs/component_gated_v1_best_by_backbone.csv`
