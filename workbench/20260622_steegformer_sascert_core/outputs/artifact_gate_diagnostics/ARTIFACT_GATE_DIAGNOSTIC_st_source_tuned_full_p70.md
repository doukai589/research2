# Artifact Gate Diagnostic

- Score tag: `st_source_tuned_full`
- Output tag: `st_source_tuned_full`
- Folds: `100`
- Artifact reject percentile: `70.0`

## Aggregate

- Mean reject rate: `0.3000`
- Mean clean reject rate: `0.2500`
- Mean BadArtifact reject rate: `1.0000`
- Mean rejected clean fraction: `0.3333`
- Mean rejected BadArtifact fraction: `0.6667`
- Mean kept BadArtifact rate: `0.0000`
- Mean SoftAR - SoftWeight Macro-F1: `-0.000113`
- Mean SoftAR - SoftWeight ECE: `-0.002625`

## Decision

`artifact_gate_overprunes_clean_or_useful_candidates`

## By Augmentation Type

| Aug Type | Reject Rate | Artifact Risk | Content Score |
|---|---:|---:|---:|
| bad_artifact | 1.0000 | 1582.1421 | 0.8747 |
| bad_content | 0.0000 | 0.2595 | 0.5113 |
| bad_physio | 0.0000 | 1.2282 | 0.4060 |
| clean | 0.2500 | 269.7701 | 0.3540 |
