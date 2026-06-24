# SAS-Cert-TFConsistency v0 Training Report

This root copy mirrors the durable report in `outputs/TRAINING_REPORT.md`.

## Summary

- Status: `completed`
- Blocked: `false`
- Leakage audit: `passed`
- Decision: `CONTINUE_TFCONSISTENCY_REPAIR`

## Key Result

- Regular pool v0 vs NaiveTF:
  - delta BAcc `+0.000454`
  - delta Macro-F1 `-0.000382`
  - delta ECE `-0.002347`
  - delta NLL `-0.026090`
  - delta Brier `-0.005533`
- Risk-mixed pool v0 vs RiskMixed NaiveTF:
  - delta BAcc `+0.002660`
  - delta Macro-F1 `+0.000194`
  - delta ECE `-0.001345`
  - delta NLL `-0.031893`
  - delta Brier `-0.005024`

## Interpretation

TFConsistency v0 improves calibration and slightly improves risk-mixed BAcc, but it does not meet the risk-mixed success criterion. The main failure is stability: subject win rate is `0.35` and seed win rate is `0.00` versus RiskMixed NaiveTF.

Full details are in `outputs/SASCERT_TFCONSISTENCY_V0_REPORT.md`.
