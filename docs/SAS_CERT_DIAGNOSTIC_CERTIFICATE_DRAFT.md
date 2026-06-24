# SAS-Cert-EEG: Diagnostic Reliability Certificates for EEG Augmentation in Few-Shot Cross-Subject Adaptation

## Abstract

EEG augmentation is often treated as a straightforward way to improve few-shot cross-subject adaptation, but augmented trials are not automatically reliable. An augmented sample may preserve superficial subject style while changing task content, violating physiological structure, or introducing artifact shortcuts. This draft presents the current evidence for SAS-Cert-EEG as a diagnostic reliability certificate for EEG augmentation candidates. On PhysioNetMI left/right motor imagery, using CBraMod and ST-EEGFormer-small as two frozen EEG foundation backbones, the original scalar SAS score fails directionally on the current synthetic mixed bad-augmentation diagnostic pool: AUC is 0.1969 for CBraMod and 0.1662 for ST-EEGFormer-small. A component-gated diagnostic rule and an artifact-gate physiology score recover strong clean-vs-bad separation within this diagnostic pool, with AUC 0.8395 and 0.9022 respectively on both backbones. However, the same evidence does not support promoting current SAS-Cert weighting or rejection policies as deployable training methods. CBraMod repaired weighting improves Macro-F1 by 4.26 percentage points but worsens ECE by 2.27 percentage points, while ST no-reject soft weighting improves mean Macro-F1 by 0.64 percentage points but fails subject and seed reliability. We therefore frame SAS-Cert as a diagnostic certificate that exposes augmentation failure modes and score-direction errors, not yet as a reliable augmentation-selection policy.

Evidence sources: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/compact_result.json`, `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`.

## 1. Introduction

Few-shot cross-subject EEG adaptation is fragile because the target subject provides only a small support set while source subjects may differ in signal scale, spectral structure, channel covariance, artifact burden, and task-related patterns. Data augmentation is a natural response to this scarcity, but it creates a reliability problem: more samples are not necessarily better samples.

The core question of this project is:

```text
How can we decide whether an augmented EEG sample is a beneficial subject-style
variation or a harmful task-content, physiology, or artifact distortion?
```

This question is deliberately different from asking whether a new augmentation policy improves average accuracy. The present results show why that distinction matters. The same certificate signals that diagnose harmful augmentation candidates can fail to produce a stable deployable training policy across subjects and seeds. A diagnostic score can be scientifically useful even when it is not yet a reliable weighting rule.

This draft therefore uses conservative wording:

- Supported claim: SAS-Cert is a diagnostic reliability certificate for EEG augmentation candidates.
- Unsupported claim: SAS-Cert is a deployable augmentation weighting or rejection method that reliably improves few-shot adaptation across target subjects and seeds.

Claim support table: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`.

## 2. Problem Definition

An EEG augmentation candidate should satisfy multiple constraints:

| Axis | Diagnostic Question | Failure Risk |
|---|---|---|
| Content | Does the sample preserve label-relevant MI content? | label/content drift |
| Style | Is the subject/session style plausible? | style mismatch |
| Physiology | Is the bandpower/covariance/topology plausible? | physiological violation |
| Artifact | Does it avoid non-neural shortcuts? | artifact contamination |

The important point is that these axes are not interchangeable. Artifact-safe scores can be excellent for bad-artifact detection while being inverted or misleading for other bad types. Content scores can also be backbone-dependent. This makes a fixed universal scalar score risky unless its direction is audited.

Failure-mode definitions are recorded in `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv`.

## 3. SAS-Cert Diagnostic Framework

SAS-Cert is treated here as a certificate profile rather than a single final score. The current diagnostic pack uses:

- `content_score`: feature similarity between original and augmented candidates;
- `style_score`: proximity to support-style statistics;
- `physio_score`: preservation of mu/beta and covariance-like structure;
- `artifact_safe_score`: inverse artifact risk from drift, burst, channel energy, and kurtosis proxies;
- `component_gated_v1`: `ranknorm(0.75 * physio_score + 0.25 * style_score) * artifact_gate_pass`;
- `score_artifact_gate_physio`: physiology score with high artifact-risk candidates gated down.

The framework requires score-direction auditing before any score is interpreted as beneficial. This is a central empirical result: on the current PhysioNetMI synthetic mixed-bad diagnostic pool, the old scalar `sas_score` is directionally wrong on both backbones.

## 4. Experimental Protocol

The locked diagnostic pack uses:

- Dataset: PhysioNetMI / EEGMMI.
- Task: left vs right motor imagery.
- Runs: R04, R08, R12.
- Backbones:
  - CBraMod frozen encoder.
  - ST-EEGFormer-small source-tuned frozen encoder.
- Target support: few-shot target support used for adaptation and candidate generation in prior workbench outputs.
- Target held-out trials: final evaluation only.
- Diagnostic pack mode: existing-output-only summarization.

No new training, no new backbone, and no new dataset were introduced in the diagnostic certificate pack. The protocol leakage audit is stored in `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`.

## 5. Results

### 5.1 Scalar Score Direction Failure

The old scalar SAS score fails on the current synthetic mixed bad-augmentation diagnostic pool:

| Backbone | Current Scalar SAS AUC |
|---|---:|
| CBraMod frozen | 0.1969 |
| ST-EEGFormer-small source-tuned | 0.1662 |

Since the AUC is computed with high score meaning clean, values far below 0.5 indicate that the scalar score is directionally wrong on this pool. This is not a small performance miss; it is a sign that scalar aggregation can invert the intended meaning of the certificate.

Source: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`.

### 5.2 Component Diagnostics Recover Clean-vs-Bad Separation

Two diagnostic variants recover strong separation on both backbones within the current diagnostic pool:

| Backbone | Component-Gated v1 AUC | Artifact-Gate Physio AUC |
|---|---:|---:|
| CBraMod frozen | 0.8395 | 0.9022 |
| ST-EEGFormer-small source-tuned | 0.8395 | 0.9022 |

This supports the diagnostic certificate claim. It also supports the move away from a single universal scalar toward a multi-component reliability profile.

Source: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`.

### 5.3 Bad-Type Evidence

The bad-type table shows that components behave differently depending on the failure mode. For example:

- `artifact_safe_score` identifies bad artifact strongly, with AUC 1.0000 for bad-artifact separation on both backbones.
- `physio_score` separates bad-content and bad-physio candidates strongly, with AUC around 0.94.
- `content_score` is backbone- and bad-type-dependent, including a BadContent conflict between CBraMod and ST.
- The old scalar `sas_score` is inverted or weak for several bad types.

Source: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/bad_type_component_auc.csv`.

### 5.4 Training Policies Are Not Promoted

The diagnostic evidence does not translate into a deployable weighting or rejection policy.

| Backbone | Policy | Delta BAcc | Delta Macro-F1 | Delta ECE | Decision |
|---|---|---:|---:|---:|---|
| CBraMod frozen | current SASCert SoftAR | +0.0124 | +0.0005 | +0.0277 | not promoted |
| CBraMod frozen | repaired artifact-gate physio | +0.0111 | +0.0426 | +0.0227 | not promoted |
| CBraMod frozen | repaired temperature scaling | +0.0111 | +0.0426 | +0.0221 | not promoted |
| ST-EEGFormer-small | SoftWeight no-reject | +0.0065 | +0.0064 | +0.0003 | not promoted |

CBraMod repaired weighting produces a large Macro-F1 gain, but the ECE increase violates the calibration requirement. ST no-reject soft weighting has a positive mean effect and nearly stable calibration, but subject/seed reliability fails. Its majority-seed subject win rate is 0.15, and its seed win rate is 0.00.

Sources: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/training_policy_summary.csv`, `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json`.

### 5.5 Utility Alignment Audit

The ST utility-alignment audit tested whether candidate-only fold summaries explain which folds benefit from SoftWeight no-reject. The strongest candidate-only feature was `clean_artifact_risk_raw_mean`, with Spearman correlation 0.1168 against SoftWeight Macro-F1 gain. This is below the actionable threshold of 0.35.

Therefore, the project parks ST weighting variants rather than designing another gate from weak retrospective correlations.

Source: `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/compact_utility_alignment_audit.json`.

## 6. Discussion

The main conclusion is that diagnostic certification and deployable augmentation selection are distinct problems.

The current data support:

```text
bad/clean augmentation is separable
  -> score direction must be audited
  -> component diagnostics can recover separation
```

The current data do not support:

```text
component diagnostics
  -> reliable training weights or rejection policies across subjects/seeds
```

This distinction is valuable. A negative or unstable training-policy result prevents overclaiming and clarifies where future work should focus. For CBraMod, the likely next hypothesis would need to be calibration-aware. For ST-EEGFormer-small, a future weighting policy would need a non-leaky support-only predictor of utility, which the current audit did not find.

## 7. Limitations

The current diagnostic pack is intentionally narrow:

- It uses one core dataset, PhysioNetMI.
- It uses two backbones, CBraMod and ST-EEGFormer-small.
- The bad augmentation pool is synthetic and defines the diagnostic task.
- Training policies are evaluated from existing workbench outputs and are not promoted.
- The diagnostic pack supports a paper route, but not a broad deployment claim.

These limitations are not treated as defects to hide. They are part of the conservative contribution: SAS-Cert is useful as a diagnostic certificate, and the training-policy problem remains open.

## 8. Conclusion

SAS-Cert-EEG is currently best supported as a diagnostic reliability certificate for EEG augmentation candidates. On PhysioNetMI, two frozen EEG foundation backbones show that the old scalar SAS score can be directionally wrong on the current synthetic mixed-bad diagnostic pool, while component-gated and artifact-gate-physio diagnostics recover strong clean-vs-bad separation within that pool. At the same time, current weighting and rejection policies are not reliable enough for promotion as deployable training methods. The paper path should therefore emphasize diagnostic reliability, score-direction auditing, and the gap between augmentation diagnosis and training utility.

## Evidence Index

- Diagnostic pack report: `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PACK_PHYSIONETMI.md`
- Evidence checklist: `docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md`
- Compact result: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/compact_result.json`
- Claim support table: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`
- Diagnostic AUC table: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`
- Training policy table: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/training_policy_summary.csv`
- Protocol audit table: `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`
