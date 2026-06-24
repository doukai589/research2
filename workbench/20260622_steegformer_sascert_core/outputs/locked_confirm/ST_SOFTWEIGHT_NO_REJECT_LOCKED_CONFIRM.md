# ST SoftWeight No-Reject Locked Confirm

## Scope

- Dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`
- Backbone: frozen source-tuned `ST-EEGFormer-small`
- Targets: `90-109`
- Seeds: `20,21,22,23,24`
- Support: `5-shot` per class
- Output tag: `st_source_tuned_full`
- Protocol note: target held-out trials are used only for final evaluation in these existing outputs.

## Mean Metrics

| Group | BAcc | Macro-F1 | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| `NaiveAug_LS010` | 0.7088 | 0.7045 | 0.2079 | 0.6853 | 0.4159 |
| `SoftWeight_noReject_LS010` | 0.7153 | 0.7109 | 0.2082 | 0.6832 | 0.4103 |
| `SASCert_SoftAR_LS010` | 0.7149 | 0.7108 | 0.2056 | 0.6810 | 0.4097 |

## Confirmation Gates

| Comparison | Delta BAcc | Delta Macro-F1 | Delta ECE | Delta NLL | Delta Brier | Positive-Mean Subject Rate | Majority-Seed Subject Win Rate | Seed Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SoftWeight - Naive` | 0.0065 | 0.0064 | 0.0003 | -0.0022 | -0.0057 | 0.6000 | 0.1500 | 0.0000 |
| `SoftWeight - SoftAR` | 0.0003 | 0.0001 | 0.0026 | 0.0022 | 0.0006 | 0.5000 | 0.0500 | 0.0000 |

## Subject Heterogeneity

Best subjects for `SoftWeight - Naive`:

| Subject | Mean Delta Macro-F1 | Seed Positive Fraction |
|---:|---:|---:|
| 106 | 0.0408 | 1.0000 |
| 108 | 0.0397 | 0.8000 |
| 98 | 0.0258 | 0.4000 |
| 95 | 0.0147 | 0.4000 |
| 94 | 0.0146 | 0.4000 |

Worst subjects for `SoftWeight - Naive`:

| Subject | Mean Delta Macro-F1 | Seed Positive Fraction |
|---:|---:|---:|
| 105 | -0.0174 | 0.2000 |
| 93 | -0.0135 | 0.2000 |
| 102 | -0.0119 | 0.2000 |
| 101 | -0.0057 | 0.0000 |
| 97 | -0.0051 | 0.2000 |

## Decision

`do_not_promote_softweight_no_reject`

SoftWeight no-reject has a real positive average signal against NaiveAug: Macro-F1 improves by about `+0.64pp`, BAcc by about `+0.65pp`, NLL and Brier improve, and ECE is nearly unchanged. However, it fails the reliability gate because gains are not stable enough across subjects and seeds.

This means the branch is a useful phenomenon and should remain the simplest ST training candidate, but it is not strong enough to become the locked main method. The next step should be a focused failure synthesis around why score/weighting gains are heterogeneous, not another unconstrained gate search.
