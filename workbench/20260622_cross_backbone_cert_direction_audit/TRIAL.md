# cross_backbone_cert_direction_audit

## Intent

Audit whether SAS-Cert component directions are stable across the two locked
backbones on the same PhysioNetMI mini scope.

This is a score-row audit only:

- No retraining
- No new augmentation generation
- No target held-out labels
- No new dataset
- No new backbone

## Scope

- Dataset: `PhysioNetMI`
- Task: left vs right motor imagery
- Targets: `90,91,92`
- Seeds: `20,21`
- Backbones:
  - `ST-EEGFormer-small_source_tuned`
  - `CBraMod_frozen`

## Inputs

- ST score rows:
  - `workbench/20260622_steegformer_sascert_core/outputs/score_rows/st_source_tuned_full/target{target}_seed{seed}.csv`
- CBraMod score rows:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/score_rows/mini/target{target}_seed{seed}.csv`

## Command

```bash
PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. \
python workbench/20260622_cross_backbone_cert_direction_audit/01_cross_backbone_cert_direction_audit.py
```

## Results

### Overall Component AUC

| Backbone | content | style | physio | artifact_safe | current sas | artifact_gate_physio |
|---|---:|---:|---:|---:|---:|---:|
| `CBraMod_frozen` | `0.3044` | `0.6408` | `0.8444` | `0.3333` | `0.1969` | `0.9022` |
| `ST-EEGFormer-small_source_tuned` | `0.2722` | `0.6408` | `0.8444` | `0.3333` | `0.1662` | `0.9022` |

### Key Bad-Type Finding

`content_score` is not stable by bad type:

| Bad type | Component | ST AUC | CBraMod AUC | Interpretation |
|---|---|---:|---:|---|
| `bad_content` | `content_score` | `0.2630` | `0.9025` | direction conflict |
| `bad_physio` | `content_score` | `0.5224` | `0.0054` | large gap, ST weak/mixed |

### Best Score-Only Variant

For both backbones, the best score-only variant is:

```text
score_artifact_gate_physio
```

with clean-vs-all-bad AUC:

- CBraMod: `0.9022`
- ST-EEGFormer-small: `0.9022`

## Decision

`revise_scalar_score_before_training_expansion`

The current fixed scalar `sas_score` should not be promoted. The evidence now
supports a multi-dimensional or component-gated certificate rule:

- physio/style are stable clean-high components on this synthetic bad pool;
- artifact-safe is reliable for BadArtifact but inverted for BadContent and
  BadPhysio under overall clean-vs-bad pooling;
- content embedding similarity is backbone/bad-type dependent and should not be
  used as a universal positive component without a bad-type gate.

## Primary Outputs

- `outputs/CROSS_BACKBONE_CERT_DIRECTION_AUDIT.md`
- `outputs/compact_cross_backbone_cert_direction_audit.json`
- `outputs/component_auc_by_backbone.csv`
- `outputs/component_auc_by_bad_type.csv`
- `outputs/direction_conflict_table.csv`
- `outputs/bad_type_direction_conflict_table.csv`
- `outputs/score_distribution_by_aug_type.csv`
- `outputs/best_variants.csv`
