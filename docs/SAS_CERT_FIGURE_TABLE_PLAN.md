# SAS-Cert Figure and Table Plan

## Purpose

This plan converts the current diagnostic-certificate evidence into a compact manuscript figure/table package. It uses only existing outputs. It does not request new experiments.

Polished draft:

```text
docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md
```

## Main Figures

### Figure 1: SAS-Cert Diagnostic Certificate Overview

Goal:

Show that an augmented EEG candidate is evaluated as a multi-component certificate profile rather than a single un-audited scalar score.

Panel plan:

- Panel A: few-shot target support, source/support augmentation candidates, held-out target evaluation separated.
- Panel B: four certificate axes: content stability, style plausibility, physiological fidelity, artifact safety.
- Panel C: direction audit and component-gated diagnostic rule.
- Panel D: claim boundary: diagnostic certificate supported; deployable weighting policy not promoted.

Evidence/source paths:

- `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv`

Caption draft:

```text
SAS-Cert-EEG evaluates each augmented EEG candidate with a multi-component
diagnostic profile. Candidate generation and ranking use only legal
source/support information; held-out target trials are reserved for final
evaluation. The certificate separates content stability, support-style
plausibility, physiological fidelity, and artifact safety, then audits score
direction before any interpretation. In the current evidence package, this
profile supports diagnostic reliability analysis but not promotion of a
deployable weighting or rejection training policy.
```

### Figure 2: Scalar Score Failure and Component Diagnostic Recovery

Goal:

Visualize the main diagnostic result: old scalar SAS fails directionally, while component-gated diagnostics recover clean-vs-bad separation.

Panel plan:

- Panel A: bar chart of current scalar SAS AUC for CBraMod and ST.
- Panel B: bar chart comparing current scalar, component-gated v1, artifact-gate physio.
- Panel C: optional dashed AUC=0.5 random line and AUC=0.70 diagnostic threshold.

Evidence/source paths:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`

Caption draft:

```text
The old scalar SAS score is directionally wrong on the current synthetic
mixed bad-augmentation diagnostic pool, with AUC below 0.5 for both locked
backbones. Component-level diagnostic rules recover strong clean-vs-bad
separation within the same pool. This supports the need for score-direction
auditing and component profiles rather than an un-audited universal scalar.
```

### Figure 3: Component Specificity by Bad-Augmentation Type

Goal:

Show why SAS-Cert should be interpreted as a profile. Components detect different failure modes.

Panel plan:

- Heatmap: rows = bad type and backbone; columns = content, style, physio, artifact-safe, old scalar, artifact-gate physio.
- Color = AUC high score means clean.
- Mark cells below 0.5 as direction conflict.

Evidence/source paths:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/bad_type_component_auc.csv`

Caption draft:

```text
Certificate components are failure-mode specific. Artifact safety separates
bad-artifact candidates strongly, while physiology is strongest for the
current bad-content and bad-physio definitions. Some components invert on
specific bad types, showing why a fixed scalar score can be misleading and
why component-level direction auditing is necessary.
```

### Figure 4: Diagnostic Success Does Not Equal Training Utility

Goal:

Make the most important negative result visible: diagnostic AUC success does not justify deployable training promotion.

Panel plan:

- Panel A: training policy deltas versus Naive for BAcc, Macro-F1, and ECE.
- Panel B: reliability annotations for ST SoftWeight: subject win rate and seed win rate.
- Panel C: decision labels: not promoted due to calibration or reliability.

Evidence/source paths:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/training_policy_summary.csv`
- `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json`

Caption draft:

```text
Diagnostic separability does not automatically translate into reliable
few-shot training utility. CBraMod repaired weighting improves Macro-F1 but
fails the calibration gate, while ST no-reject soft weighting has a positive
mean effect but fails subject/seed reliability. These results justify
framing SAS-Cert as a diagnostic certificate rather than a deployable
augmentation-selection policy.
```

### Figure 5: Causal Chain and Failure Point

Goal:

Summarize the causal interpretation in one conceptual diagram.

Panel plan:

```text
bad/clean augmentation separable
  -> score direction auditing required
  -> component diagnostics recover separation
  -X-> stable deployable weights/rejection across subjects/seeds
```

Evidence/source paths:

- `docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`

Caption draft:

```text
The current evidence supports the diagnostic portion of the causal chain but
breaks before deployable training utility. This failure point is the central
reason the manuscript should avoid claiming reliable augmentation-selection
improvement and instead emphasize diagnostic certification.
```

## Main Tables

### Table 1: Protocol and Claim Boundary

Rows:

- Dataset, task, runs.
- Backbones.
- Candidate pool and target support.
- Held-out target usage.
- Supported claims.
- Unsupported claims.

Source paths:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`

### Table 2: Diagnostic AUC Summary

Rows:

- CBraMod frozen.
- ST-EEGFormer-small source-tuned.

Columns:

- current scalar SAS AUC.
- component-gated v1 AUC.
- artifact-gate physio AUC.
- physio score AUC.
- style score AUC.
- content score AUC.
- artifact-safe score AUC.

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`

### Table 3: Bad-Type Component AUC

Rows:

- `bad_artifact`, `bad_content`, `bad_physio` for each backbone.

Columns:

- content.
- style.
- physio.
- artifact-safe.
- old scalar.
- artifact-gate physio.

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/bad_type_component_auc.csv`

### Table 4: Training Policy Non-Promotion

Rows:

- CBraMod current SoftAR.
- CBraMod repaired artifact-gate physio.
- CBraMod repaired temperature scaled.
- ST SoftWeight no-reject.

Columns:

- scope.
- Delta BAcc.
- Delta Macro-F1.
- Delta ECE.
- non-promotion reason.

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/training_policy_summary.csv`

### Table 5: Claim Support and Prohibited Claims

Rows:

- C1-C10 from claim support table.

Columns:

- claim.
- stance.
- evidence summary.
- source path.
- allowed wording.

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`

## Appendix Figures and Tables

### Appendix A: Protocol Leakage Audit

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`

Use:

Document that held-out target data were not used for scoring, ranking, threshold selection, or training-policy selection in the diagnostic pack.

### Appendix B: Number Traceability

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/number_traceability.csv`

Use:

List every numerical claim and its file source.

### Appendix C: Failure-Mode Definitions

Source:

- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv`

Use:

Clarify the current synthetic bad-augmentation definitions and avoid broad claims about all real-world artifact/content/physiology failures.

## Do Not Plot As Main Evidence

- Do not plot MIRepNet reproduction results as main SAS-Cert evidence.
- Do not plot old BCIC-IV-2a EEGNet MVE as if it were the same protocol.
- Do not plot shadow training policies as promoted methods.
- Do not make a leaderboard figure; this manuscript is about diagnostic reliability, not model ranking.
