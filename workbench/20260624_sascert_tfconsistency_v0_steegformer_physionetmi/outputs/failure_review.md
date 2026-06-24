# SAS-Cert-TFConsistency v0 Failure Review

The run is not blocked and is not a catastrophic failure, but it does not meet
the stated success criteria.

## What Worked

- The route is diagnostically ordered:
  - supervised views have very low raw CE and near-perfect correctness.
  - consistency views remain mostly correct.
  - quarantined views have much higher raw CE and lower correctness.
- Calibration improved versus NaiveTF in both regular and risk-mixed pools.
- In the risk-mixed pool, v0 beat AugMixTF on BAcc and Macro-F1.

## What Failed

- Risk-mixed v0 vs RiskMixed NaiveTF:
  - delta BAcc was only `+0.002660`, below the `+0.005` criterion.
  - delta Macro-F1 was only `+0.000194`.
  - subject win rate was `0.35`, below `0.50`.
  - seed win rate was `0.00`.
- The route is conservative:
  - regular quarantine ratio: `0.6122`.
  - risk-mixed quarantine ratio: `0.6456`.
- Wrong-class frequency mixup detection is weak:
  - diagnostic AUC `0.550457`.

## Practical Conclusion

TFConsistency is a plausible algorithm branch, but v0 does not yet provide
reliable augmentation utilization. The next repair should improve content-risk
detection and route calibration rather than adding new tools or switching
backbones.
