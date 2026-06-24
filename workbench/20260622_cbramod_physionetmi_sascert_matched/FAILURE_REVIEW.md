# CBraMod PhysioNetMI Failure Review

## 1. Facts Only

| Item | Value |
|---|---:|
| Mini targets | `90,91,92` |
| Mini seeds | `20,21` |
| NaiveAug BAcc / Macro-F1 / ECE | `0.5349` / `0.4983` / `0.0843` |
| SASCert SoftAR BAcc / Macro-F1 / ECE | `0.5473` / `0.4988` / `0.1120` |
| SASCert SoftAR - Naive BAcc | `0.0124` |
| SASCert SoftAR - Naive Macro-F1 | `0.0005` |
| SASCert SoftAR - Naive ECE | `0.0277` |
| NoAug BAcc / Macro-F1 / ECE | `0.5424` / `0.5159` / `0.0817` |
| Source-only target BAcc / Macro-F1 / ECE | `0.5068` / `0.5001` / `0.0043` |
| RepairedSoftWeight - Naive Macro-F1 / ECE | `0.0426` / `0.0227` |

Data quality notes:

- No NaN/Inf was reported in the frozen feature cache.
- No raw EEG data or raw augmented arrays were copied into this workbench.
- Target held-out labels were used only for final evaluation.

## 2. Expected vs Actual

| Expectation | Actual Result | Verdict | Key Data |
|---|---|---|---|
| CBraMod features should support target few-shot adaptation above chance | Target metrics are near chance | rejected | Naive Macro-F1 `0.4983`, NoAug Macro-F1 `0.5159` |
| SASCert SoftAR should improve Macro-F1 over NaiveAug | Macro-F1 gain is negligible | rejected | delta Macro-F1 `0.0005` |
| SASCert should not worsen calibration beyond +0.01 ECE | ECE worsens beyond gate | rejected | delta ECE `0.0277` |
| Artifact gate should reject artifact candidates more than clean candidates | Gate is selective for BadArtifact | supported | BadArtifact p90 reject `0.5000`, clean reject `0.0000` |
| Cert scores should separate clean from bad candidates | Current total score is directionally wrong on the mixed bad pool | rejected | overall SAS AUC `0.1969` |

## 3. Causal Chain

```text
Bad/clean augmentation separable
→ cert score ranks useful samples higher
→ soft weighting/rejection changes training signal
→ few-shot target adaptation improves
→ calibration does not degrade
```

| Link | Status | Evidence |
|---|---|---|
| Bad/clean augmentation separable | supported by some components, rejected by current total score | physio AUC `0.8444`, current SAS AUC `0.1969` |
| Cert score ranks useful samples higher | rejected for current total score | current SAS AUC `0.1969`, direction-fixed total AUC `0.8911` |
| Soft weighting/rejection changes training signal | supported | SASCert SoftAR changes BAcc by `0.0124` and ECE by `0.0277` |
| Few-shot target adaptation improves | rejected for Macro-F1 | Macro-F1 delta `0.0005` |
| Calibration does not degrade | rejected | ECE delta `0.0277` |

First broken link:

```text
current mixed-bad SAS score direction is wrong on PhysioNetMI
→ selected/weighted augmentations do not improve Macro-F1
→ the weak frozen CBraMod feature space leaves little margin for recovery
```

## 4. Possible Explanations

1. Highest credibility: the current CBraMod PhysioNetMI certificate formula is directionally wrong for the mixed bad pool.
   - Explains: current SAS clean-vs-bad AUC is `0.1969`, while a direction-fixed score-only variant reaches `0.8911`.
   - Does not fully explain: why absolute target adaptation is weak.
   - Distinguishing measurement: train only one repaired score variant on the same mini targets after freezing the formula from score-only diagnostics.

2. Medium credibility: frozen CBraMod pooled features are not aligned enough for this PhysioNetMI protocol.
   - Explains: Source-only, NoAug, and NaiveAug are all near chance.
   - Does not fully explain: why SASCert can still lift BAcc slightly.
   - Distinguishing measurement: compare frozen pooled features against a source-tuned CBraMod or a different pooling/head policy on the same mini targets.

3. Lower credibility: the SAS artifact/content gate detects only artifact synthetic badness, not target utility.
   - Explains: artifact reject stats are plausible while Macro-F1 gains are negligible.
   - Does not explain: low absolute baseline.
   - Distinguishing measurement: correlate candidate scores with leave-one-candidate-out support validation utility, using only target support.

## 5. One Next Experiment

Do not expand the current mini matrix to full targets yet.

Next focused experiment:

```text
CBraMod cert-direction repair mini:
freeze a repaired score from existing score-only diagnostics
and run only NoAug / NaiveAug / repaired SoftWeight on targets 90-92 and seeds 20-21.
```

Diagnostic rerun result:

- The repaired score rescued classification on the mini matrix, but it still failed the calibration gate.
  RepairedSoftWeight vs Naive: Macro-F1 `0.0426`, ECE `0.0227`.


Go/No-Go:

- Go to broader CBraMod validation only if repaired SoftWeight beats both NoAug and NaiveAug by at least `+1pp` Macro-F1 without ECE worsening beyond `+0.01`.
- If the repaired score still fails, park CBraMod PhysioNetMI and keep ST-EEGFormer-small as the active PhysioNetMI backbone.

## 6. Decision

`revise_cert_calibration`

This is not a reason to abandon SAS-Cert. It says the current CBraMod
PhysioNetMI total certificate is directionally wrong under the mixed bad
candidate pool, and the frozen feature space is weak enough that bad scoring
cannot be rescued by training. The repaired score mini shows classification
can improve after direction repair, but calibration still violates the gate.
Do not run full CBraMod PhysioNetMI until a calibration-aware repaired score or
loss is defined.
