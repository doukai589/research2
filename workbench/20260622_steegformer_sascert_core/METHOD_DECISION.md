# ST-EEGFormer Method Decision

## Scope

- Backbone: `ST-EEGFormer-small`
- Dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`
- Feature extractor: source-tuned checkpoint `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Support: target subject 5-shot per class

## Result Snapshot

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7088 | 0.7045 | 0.2079 | 0.6853 | 0.4159 |
| ArtifactReject_LS010 | 0.7107 | 0.7064 | 0.2112 | 0.6854 | 0.4165 |
| SoftWeight_noReject_LS010 | 0.7153 | 0.7109 | 0.2082 | 0.6832 | 0.4103 |
| SASCert_SoftAR_LS010 | 0.7149 | 0.7108 | 0.2056 | 0.6810 | 0.4097 |

## Paired Comparisons

The table below uses the stricter majority-seed subject win definition used by
the pair summarizer.

| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Subject Win Rate | Seed Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| SoftWeight_noReject vs Naive | +0.0065 | +0.0064 | +0.0003 | -0.0022 | 0.15 | 0.00 |
| SASCert_SoftAR vs Naive | +0.0061 | +0.0063 | -0.0023 | -0.0044 | 0.25 | 0.00 |
| SoftWeight_noReject vs SoftAR | +0.0003 | +0.0001 | +0.0026 | +0.0022 | 0.05 | 0.00 |

## Artifact Gate Diagnostic

| Gate Percentile | Clean Reject | BadArtifact Reject | Training Finding |
|---:|---:|---:|---|
| p90 | 0.00 | 0.50 | Conservative, best calibration/NLL when paired with SoftAR |
| p80 | 0.00 | 1.00 | Mini training hurts BAcc/Macro-F1 despite cleaner detection |
| p70 | 0.25 | 1.00 | Rejects clean samples, reject |

## Subject Heterogeneity

Subject-mean aggregation gives a less pessimistic but still incomplete picture:

| Method vs Naive | Mean Subject Delta Macro-F1 | Subject-Mean Win Rate |
|---|---:|---:|
| SoftWeight_noReject_LS010 | +0.0064 | 0.60 |
| SASCert_SoftAR_LS010 | +0.0063 | 0.65 |

SoftWeight gains correlate negatively with baseline Macro-F1 (`r=-0.4209`),
which suggests the method helps weaker target subjects more than already-strong
ones.

## Decision

```text
do_not_promote_any_st_method_yet
```

Mean effects are positive, but reliability is conditional on subject regime.
The next useful question is now routing, not another artifact threshold tweak:

```text
Can support-only features identify when to use NaiveAug, SoftWeight_noReject,
or SASCert_SoftAR without looking at target-test labels?
```

## Next Action

Develop a validation-subject routing rule:

- Config: `configs/experiments/steegformer_physionetmi_support_routing_dev.yaml`
- Dev subjects: `71-89`
- Final subjects remain untouched for rule development: `90-109`
- Allowed inputs: support-only features and candidate scores
- Forbidden inputs: final target-test labels, final target-test metrics, subject-id shortcuts

No new dataset, backbone, or broad hyperparameter search should be added before this routing-dev step.
