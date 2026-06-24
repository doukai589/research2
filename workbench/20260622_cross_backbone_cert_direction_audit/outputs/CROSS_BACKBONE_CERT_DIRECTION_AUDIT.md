# Cross-Backbone Certificate Direction Audit

## Scope

- Dataset: `PhysioNetMI`
- Targets: `90,91,92`
- Seeds: `20,21`
- Backbones: `ST-EEGFormer-small_source_tuned`, `CBraMod_frozen`
- Inputs: existing score rows only; no retraining and no new augmentation generation.

## Component AUC

| Backbone | Component | AUC high=clean | Direction |
|---|---|---:|---|
| CBraMod_frozen | content_score | 0.3044 | bad_high_or_inverted |
| CBraMod_frozen | style_score | 0.6408 | clean_high |
| CBraMod_frozen | physio_score | 0.8444 | clean_high |
| CBraMod_frozen | artifact_safe_score | 0.3333 | bad_high_or_inverted |
| CBraMod_frozen | sas_score | 0.1969 | bad_high_or_inverted |
| CBraMod_frozen | score_artifact_gate_physio | 0.9022 | clean_high |
| ST-EEGFormer-small_source_tuned | content_score | 0.2722 | bad_high_or_inverted |
| ST-EEGFormer-small_source_tuned | style_score | 0.6408 | clean_high |
| ST-EEGFormer-small_source_tuned | physio_score | 0.8444 | clean_high |
| ST-EEGFormer-small_source_tuned | artifact_safe_score | 0.3333 | bad_high_or_inverted |
| ST-EEGFormer-small_source_tuned | sas_score | 0.1662 | bad_high_or_inverted |
| ST-EEGFormer-small_source_tuned | score_artifact_gate_physio | 0.9022 | clean_high |

## Direction Conflicts

| Component | ST AUC/Dir | CBraMod AUC/Dir | Conflict |
|---|---:|---:|---|
| artifact_safe_score | 0.3333 / bad_high_or_inverted | 0.3333 / bad_high_or_inverted | False |
| content_score | 0.2722 / bad_high_or_inverted | 0.3044 / bad_high_or_inverted | False |
| physio_score | 0.8444 / clean_high | 0.8444 / clean_high | False |
| sas_score | 0.1662 / bad_high_or_inverted | 0.1969 / bad_high_or_inverted | False |
| score_artifact_gate_physio | 0.9022 / clean_high | 0.9022 / clean_high | False |
| style_score | 0.6408 / clean_high | 0.6408 / clean_high | False |

## Bad-Type Direction Conflicts

| Bad type | Component | ST AUC/Dir | CBraMod AUC/Dir | Conflict |
|---|---|---:|---:|---|
| bad_content | content_score | 0.2630 / bad_high_or_inverted | 0.9025 / clean_high | True |
| bad_physio | content_score | 0.5224 / weak_or_mixed | 0.0054 / bad_high_or_inverted | False |

## Best Variants

| Backbone | Best variant | Best AUC | Current SAS AUC | Current SAS direction |
|---|---|---:|---:|---|
| CBraMod_frozen | score_artifact_gate_physio | 0.9022 | 0.1969 | bad_high_or_inverted |
| ST-EEGFormer-small_source_tuned | score_artifact_gate_physio | 0.9022 | 0.1662 | bad_high_or_inverted |

## Interpretation

- The existing total `sas_score` is directionally wrong on both backbones in the matched PhysioNetMI mini scope.
- The most useful score-only variant on both backbones is `score_artifact_gate_physio`.
- The strongest component-level backbone difference is `content_score` on `bad_content`: CBraMod treats clean as high, while ST treats bad content as high.
- Component directions must be inspected by bad type; overall clean-vs-bad AUC hides important failure modes.
- This supports the revised scientific framing: SAS-Cert should be treated as a multi-dimensional certificate, not a fixed universal scalar score.

## Decision

`revise_scalar_score_before_training_expansion`

Next action: define a backbone-aware or component-gated certificate rule from component directions, then test it in a small ST reliability setting before any new full expansion.
