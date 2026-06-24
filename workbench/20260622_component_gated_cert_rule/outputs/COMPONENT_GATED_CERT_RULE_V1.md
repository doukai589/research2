# Component-Gated Certificate Rule v1

## Rule

```text
artifact_gate_pass = artifact_risk < fold_p90
base = 0.75 * physio_score + 0.25 * style_score
component_gated_v1 = ranknorm(base) * artifact_gate_pass
component_gated_v1_soft = ranknorm(base) * (0.2 + 0.8 * artifact_safe_rank)
content_score is diagnostic only and is not a positive term in v1
```

## Score-Only AUC

| Backbone | Current SAS | ArtifactGatePhysio | ComponentGated v1 | ComponentGated v1 soft | Best |
|---|---:|---:|---:|---:|---|
| CBraMod_frozen | 0.1969 | 0.9022 | 0.8395 | 0.7383 | score_artifact_gate_physio `0.9022` |
| ST-EEGFormer-small_source_tuned | 0.1662 | 0.9022 | 0.8395 | 0.7383 | score_artifact_gate_physio `0.9022` |

## Gate Stats

| Backbone | Aug type | Artifact reject rate | Content warning rate |
|---|---|---:|---:|
| CBraMod_frozen | bad_artifact | 0.5000 | 0.5333 |
| CBraMod_frozen | bad_content | 0.0000 | 0.8667 |
| CBraMod_frozen | bad_physio | 0.0000 | 0.7667 |
| CBraMod_frozen | clean | 0.0000 | 0.2167 |
| ST-EEGFormer-small_source_tuned | bad_artifact | 0.5000 | 0.9333 |
| ST-EEGFormer-small_source_tuned | bad_content | 0.0000 | 0.3500 |
| ST-EEGFormer-small_source_tuned | bad_physio | 0.0000 | 0.6167 |
| ST-EEGFormer-small_source_tuned | clean | 0.0000 | 0.3500 |

## Decision

`component_gated_v1_defined_score_validated`

The v1 rule fixes the obvious scalar-score direction failure, but it does not beat the simpler `score_artifact_gate_physio` diagnostic score. Use v1 as the interpretable certificate policy, and treat `score_artifact_gate_physio` as the current strongest score-only baseline for the next small ST reliability test.
