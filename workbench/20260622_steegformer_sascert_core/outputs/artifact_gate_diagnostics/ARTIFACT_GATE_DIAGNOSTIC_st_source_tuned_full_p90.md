# Artifact Gate Diagnostic

- Score tag: `st_source_tuned_full`
- Output tag: `st_source_tuned_full`
- Folds: `100`
- Artifact reject percentile: `90.0`

## Aggregate

- Mean reject rate: `0.1000`
- Mean clean reject rate: `0.0000`
- Mean BadArtifact reject rate: `0.5000`
- Mean rejected clean fraction: `0.0000`
- Mean rejected BadArtifact fraction: `1.0000`
- Mean kept BadArtifact rate: `0.5000`
- Mean SoftAR - SoftWeight Macro-F1: `-0.000113`
- Mean SoftAR - SoftWeight ECE: `-0.002625`

## Decision

`artifact_gate_precise_but_conservative`

## By Augmentation Type

| Aug Type | Reject Rate | Artifact Risk | Content Score |
|---|---:|---:|---:|
| bad_artifact | 0.5000 | 1582.1421 | 0.8747 |
| bad_content | 0.0000 | 0.2595 | 0.5113 |
| bad_physio | 0.0000 | 1.2282 | 0.4060 |
| clean | 0.0000 | 269.7701 | 0.3540 |
