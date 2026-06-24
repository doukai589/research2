# SAS-Cert-EEG Research Roadmap

## Scientific Question

The project is centered on this question:

> In few-shot cross-subject adaptation of EEG foundation models, how can we decide whether an augmented EEG sample is a beneficial subject-style variation or a harmful task-content/physiology/artifact distortion?

The contribution should be framed as a reliability and diagnosis mechanism for augmented EEG samples, not as a pure accuracy-chasing method.

## Locked Main Scope

Current long-term core:

- Backbone 1: `CBraMod`
- Backbone 2: `ST-EEGFormer-small`
- Dataset: `PhysioNetMI`
- Task: left vs right motor imagery, runs `R04/R08/R12`
- Method policy:
  - CBraMod anchor branch: completed matched mini diagnostics; full expansion parked because calibration repair failed.
  - ST-EEGFormer branch: weighting variants parked after the focused utility-alignment audit.
  - Cross-backbone certificate direction audit and component-gated rule definition completed; the component-gated mini training test did not justify expansion.

Secondary / parked:

- `LaBraM-base`: secondary reliability/calibration baseline.
- `EEGPT-large4E`: paused until adapter/runtime repair is explicitly needed.
- `MIRepNet`: parked outside the main scientific route.
- `BCIC-IV-2a`: completed historical SAS-Cert anchor, not the next active dataset.

## Why This Scope

PhysioNetMI is the best next dataset because:

- It is already locally audited and runnable.
- ST-EEGFormer-small showed strong full fine-tune performance: BAcc `0.7669`.
- CBraMod has a stable paper-code reference on PhysioNetMI: BAcc `0.6285`, close to the paper target.
- This dataset allows a clean two-backbone comparison on one task without expanding the problem too early.

CBraMod and ST-EEGFormer-small are the two main backbones because:

- CBraMod is the completed SAS-Cert anchor.
- ST-EEGFormer-small is the strongest current companion EEG-FM on PhysioNetMI.
- Their performance gap is scientifically useful: SAS-Cert should be tested on both a stable anchor and a stronger representation model.

## Active Research Steps

### Step 1: Stabilize Shared Infrastructure

Promote only reusable code from workbench into `sas_core/`:

- PhysioNetMI loader and split protocol
- ST-EEGFormer wrapper
- CBraMod wrapper if needed for PhysioNetMI few-shot adaptation
- metrics and calibration utilities
- cert scoring and weighting utilities

### Step 2: ST-EEGFormer-small + PhysioNetMI SAS-Cert Validation

Groups:

- `NaiveAug_LS010`
- `ArtifactReject_LS010`
- `SoftWeight_noReject_LS010`
- `SASCert_SoftAR_LS010`

Completed comparison:

- `SASCert_SoftAR_LS010` vs `NaiveAug_LS010`
- `SoftWeight_noReject_LS010` vs `NaiveAug_LS010`

Original success target:

- Macro-F1 improves by at least `+0.5pp`
- Balanced accuracy does not meaningfully drop
- ECE/NLL/Brier do not worsen beyond `+0.01`
- Subject/seed win rate on Macro-F1 is at least `0.65`

Current result:

- Source-tuned ST features repaired the near-chance feature-space problem.
- `SASCert_SoftAR_LS010` and `SoftWeight_noReject_LS010` both improved mean metrics over `NaiveAug_LS010`.
- Neither method met subject/seed reliability criteria.
- `SoftWeight_noReject_LS010` was the best mean classification group, while `SASCert_SoftAR_LS010` was better for ECE/NLL/Brier.

Completed diagnostic:

- Subject-level heterogeneity analysis found that `SoftWeight_noReject_LS010` has positive mean Macro-F1 delta but only `0.60` subject win rate.
- `SASCert_SoftAR_LS010` reached `0.65` subject win rate in subject-mean aggregation but failed seed reliability and was slightly below no-reject soft weighting for mean classification.
- SoftWeight gains were negatively correlated with baseline Macro-F1 (`r=-0.42`), suggesting it helps weaker target subjects more than already-strong ones.

Support-routing dev result:

- On validation subjects `71-89`, the best constant strategy was `SASCert_SoftAR_LS010` with Macro-F1 `0.6654`.
- A full-dev threshold rule could fit Macro-F1 `0.6675`, but leave-one-subject-out routing dropped to Macro-F1 `0.6594`.
- Decision: do not freeze a ST routing rule and do not apply routed ST methods to final targets.

Component-gated mini training result:

- Scope: targets `90,91,92`, seeds `20,21`, frozen source-tuned ST features.
- `SoftWeight_noReject_LS010` remained the best mean training group:
  - BAcc `0.7902`
  - Macro-F1 `0.7898`
  - ECE `0.1471`
  - NLL `0.4908`
- `ComponentGatedV1_LS010` did not improve training utility:
  - BAcc `0.7823`
  - Macro-F1 `0.7817`
  - ECE `0.1473`
  - NLL `0.5093`
  - Delta vs Naive Macro-F1 `-0.46pp`
  - Delta vs SoftWeight no-reject Macro-F1 `-0.81pp`
- `ArtifactGatePhysio_LS010` matched `ComponentGatedV1_LS010` on BAcc/Macro-F1
  and did not beat Naive.

Interpretation:

- Score-only clean-vs-bad AUC is not sufficient proof of training usefulness.
- The gate can be a good diagnostic detector while still over-constraining the
  augmentation set or misaligning the weights used by the classifier head.

Current next step:

- Do not expand component-gated or artifact-gate-physio on ST.
- `SoftWeight_noReject_LS010` has now been packaged and confirmed as a positive
  mean but unreliable effect:
  - Delta Macro-F1 vs Naive: `+0.64pp`
  - Delta BAcc vs Naive: `+0.65pp`
  - Delta ECE vs Naive: `+0.03pp`
  - Positive subject mean-delta rate: `0.60`
  - Majority-seed subject win rate: `0.15`
  - Seed win rate: `0.00`
- Decision: `do_not_promote_softweight_no_reject`.
- Next step is exactly one existing-output utility-alignment audit:
  `support_candidate_utility_alignment_audit`.

SoftWeight failure synthesis:

- First broken link:

```text
clean-vs-bad certificate quality
  → subject/seed-stable training utility
```

- Main interpretation: the certificate can be diagnostically meaningful while
  still not being a deployable weighting rule for every target subject.
- Next audit should test whether support/candidate features explain which
  subjects benefit. If not, park ST weighting variants and frame SAS-Cert as
  diagnostic certification rather than reliable deployable selection.

Utility-alignment audit result:

- Inputs: existing `st_source_tuned_full` metrics and score rows only.
- No new training, dataset, or backbone.
- Candidate-only fold features did not explain SoftWeight benefit:
  - strongest absolute Spearman: `0.1168`
  - strongest feature: `clean_artifact_risk_raw_mean`
  - threshold for actionable alignment: `0.35`
- Decision: `park_st_weighting_variants`.

Updated interpretation:

- SAS-Cert diagnostics are meaningful: they expose bad augmentation modes,
  score direction errors, and artifact/physiology risk.
- Current weighting/rejection policies are not reliable enough to be promoted
  as deployable training methods.
- The next step is not another gate search. It is a method reframe:
  `sas_cert_diagnostic_reframe_after_weighting_failures`.

Diagnostic certificate pack result:

- Output report: `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PACK_PHYSIONETMI.md`.
- Compact result:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/compact_result.json`.
- All gates passed:
  - diagnostic AUC gate: `true`
  - scalar failure gate: `true`
  - protocol gate: `true`
  - weighting-policy non-promotion gate: `true`
- Decision: `write_diagnostic_certificate_paper_path`.

Current paper-path artifact:

- `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PAPER_OUTLINE.md`
- `docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md`

Current active next:

- Review the first conservative paper draft for overclaiming, traceability, and
  evidence gaps:
  `configs/experiments/diagnostic_certificate_draft_review_plan.yaml`.

Current draft artifact:

- `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`

Draft review result:

- Review report: `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_REVIEW.md`.
- Revision checklist:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/draft_revision_checklist.csv`.
- Decision: `draft_review_passed_with_minor_scope_revision`.
- Revision applied: diagnostic AUC claims were scoped to the current synthetic
  mixed bad-augmentation diagnostic pool.

Current active next:

- Create a venue-agnostic LaTeX paper package from the current diagnostic
  manuscript assets:
  `configs/experiments/venue_specific_formatting_plan.yaml`.

Polish/citation package result:

- Polished draft:
  `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md`.
- Citation plan:
  `docs/SAS_CERT_CITATION_PLAN.md`.
- Figure/table plan:
  `docs/SAS_CERT_FIGURE_TABLE_PLAN.md`.
- The package keeps the conservative boundary:
  diagnostic certificate supported; deployable weighting/rejection policy not
  promoted.

Manuscript package review result:

- Review report:
  `docs/SAS_CERT_DIAGNOSTIC_MANUSCRIPT_PACKAGE_REVIEW.md`.
- Decision:
  `proceed_to_bibliography_and_figure_generation`.
- Passed:
  citation placeholder mapping, figure/table source traceability, claim safety,
  no promotion of weighting/rejection policy.
- Submission-preparation gaps:
  BibTeX resolution, removal of project-management citation note from the
  abstract, and figure/table rendering from existing CSV/JSON evidence.

Bibliography and figure/table generation result:

- Generator:
  `scripts/generate_diagnostic_manuscript_assets.py`.
- Submission-clean draft:
  `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md`.
- Bibliography trace:
  `docs/SAS_CERT_BIBLIOGRAPHY_TRACE.md`.
- Manifest:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_asset_manifest.json`.
- Generated table drafts:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/`.
- Generated figure drafts:
  `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/`.
- Still no new experiments, no new claims, and no promotion of weighting policy.

Submission package quality review result:

- Review report:
  `docs/SAS_CERT_SUBMISSION_PACKAGE_QUALITY_REVIEW.md`.
- Decision:
  `proceed_to_bibtex_latex_figure_polish`.
- Verified:
  all manifest outputs exist, 5 SVG files parse, 6 table drafts exist, internal
  project paths were removed from the submission draft, and no deployable
  weighting-policy claim was introduced.
- Remaining gaps are production-only:
  BibTeX resolution, ST-EEGFormer-small formal source record, LaTeX equation
  conversion, and figure/table styling.

BibTeX / LaTeX / figure-table polish result:

- Generator:
  `scripts/generate_diagnostic_manuscript_assets.py`.
- LaTeX-ready submission draft:
  `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT_LATEX.md`.
- BibTeX file:
  `docs/SAS_CERT_REFERENCES.bib`.
- Bibliography resolution report:
  `docs/SAS_CERT_BIBLIOGRAPHY_RESOLUTION_REPORT.md`.
- Figure/table polish report:
  `docs/SAS_CERT_FIGURE_TABLE_POLISH_REPORT.md`.
- Verification:
  13 BibTeX entries, balanced braces, all manifest outputs present, 5 SVGs
  parse, no unresolved bracket-style citation placeholders in the LaTeX draft.
- Still no new experiments, no new claims, and no promoted weighting policy.

Submission-readiness review result:

- Review report:
  `docs/SAS_CERT_SUBMISSION_READINESS_REVIEW.md`.
- Decision:
  `proceed_to_venue_specific_formatting`.
- Verified:
  all manuscript citations resolve to BibTeX entries, BibTeX braces are
  balanced, all manifest files exist, 5 SVGs parse, 6 tables exist, and the
  LaTeX draft preserves the diagnostic-only claim boundary.
- Remaining work:
  template conversion, related-work expansion using resolved citations, unused
  BibTeX cleanup, and figure/table layout.

Cross-backbone direction audit result:

- Inputs: existing score rows only, no retraining or new augmentation generation.
- Scope: targets `90,91,92`, seeds `20,21`, `ST-EEGFormer-small_source_tuned` and `CBraMod_frozen`.
- Current scalar `sas_score` is directionally wrong on both backbones:
  - CBraMod AUC `0.1969`
  - ST AUC `0.1662`
- Best score-only diagnostic variant on both backbones is `score_artifact_gate_physio`:
  - CBraMod AUC `0.9022`
  - ST AUC `0.9022`
- Stable clean-high components in this synthetic bad pool:
  - `physio_score`: AUC `0.8444` on both
  - `style_score`: AUC `0.6408` on both
- Context-dependent components:
  - `artifact_safe_score` is useful for BadArtifact but inverted for BadContent/BadPhysio when treated as a universal positive score.
  - `content_score` has a bad-type/backbone conflict for BadContent: ST AUC `0.2630`, CBraMod AUC `0.9025`.

Decision:

- `revise_scalar_score_before_training_expansion`
- SAS-Cert should be treated as a multi-dimensional / component-gated certificate, not a fixed universal scalar score.

Component-gated rule v1:

```text
artifact_gate_pass = artifact_risk < fold_p90
base = 0.75 * physio_score + 0.25 * style_score
component_gated_v1 = ranknorm(base) * artifact_gate_pass
content_score = diagnostic warning only
```

Score-only validation:

- Current scalar `sas_score`:
  - CBraMod AUC `0.1969`
  - ST AUC `0.1662`
- `component_gated_v1`:
  - CBraMod AUC `0.8395`
  - ST AUC `0.8395`
- `score_artifact_gate_physio`:
  - CBraMod AUC `0.9022`
  - ST AUC `0.9022`

Rule decision:

- `component_gated_v1_defined_score_validated`
- Use `component_gated_v1` as an interpretable diagnostic policy, not a
  promoted ST training method.
- Use `score_artifact_gate_physio` as the strongest score-only baseline, but do
  not expand it on ST without a new training-utility hypothesis.

### Step 3: CBraMod + PhysioNetMI Matched Validation

Purpose:

- Test whether the SAS-Cert effect holds on the same dataset with the anchor backbone.
- Avoid comparing CBraMod on BCIC-IV-2a against ST on PhysioNetMI as if that were a controlled study.

Current result:

- Smoke and frozen feature cache passed.
- Current `SASCert_SoftAR_LS010` mini matrix did not justify full expansion: Macro-F1 delta vs Naive was only `+0.05pp`, ECE worsened `+2.77pp`.
- Failure review found the current mixed-bad SAS score direction is wrong on PhysioNetMI:
  - current SAS clean-vs-bad AUC `0.1969`
  - direction-fixed total AUC `0.8911`
  - artifact-gate physio AUC `0.9022`
- Repaired artifact-gate physio soft weighting rescued classification:
  - Macro-F1 delta vs Naive `+4.26pp`
  - BAcc delta vs Naive `+1.11pp`
  - subject win rate `0.6667`
  - seed win rate `1.0000`
- Calibration remained outside the gate:
  - repaired ECE delta `+2.27pp`
  - support-only temperature scaling still had ECE delta `+2.21pp` and worsened NLL/Brier.

Decision:

- `park_cbramod_physionetmi_full_expansion`
- Keep the repaired-score result as evidence that certificate component direction can depend on backbone/dataset.
- Do not run full CBraMod PhysioNetMI until a new calibration hypothesis exists.

### Step 4: Decide Whether to Expand

Only expand after Step 2 diagnostic and Step 3:

- If both backbones support SAS-Cert: move to emotion dataset.
- If only CBraMod supports it: inspect ST cert scores and embedding definitions.
- If only ST supports it: revise the interpretation toward representation-dependent cert behavior.
- If neither supports it: run the failure review protocol before adding new experiments.

Current expansion decision:

- Do not add new datasets yet.
- Do not run CBraMod full expansion yet.
- Do not expand component-gated ST training.
- Park ST weighting variants after the failed utility-alignment audit.
- Reframe SAS-Cert toward diagnostic reliability certification before any new
  training experiment.
- The diagnostic certificate pack passed; proceed toward paper evidence
  packaging before adding any new training experiment.
- The venue-agnostic LaTeX paper package has now been generated:
  - `paper/sas_cert_diagnostic_certificate/main.tex`
  - `paper/sas_cert_diagnostic_certificate/references.bib`
  - `docs/SAS_CERT_VENUE_FORMATTING_REPORT.md`
- The next active step is not a new experiment. It is a package integrity and
  venue-template readiness review:
  - `configs/experiments/venue_package_integrity_review.yaml`
- The venue package integrity review has passed with an environment limitation:
  local LaTeX compile tools are unavailable here.
- Next active step:
  - `configs/experiments/venue_template_selection_plan.yaml`
- This next step should choose a concrete template route, likely starting with
  an arXiv-style article package unless the user selects a specific venue.
- Venue route selected:
  - `arxiv_article_first`
- Generated venue package:
  - `paper/sas_cert_diagnostic_certificate_venue/main.tex`
  - `docs/SAS_CERT_VENUE_SELECTION_REPORT.md`
- Current blocker for a clean upload/compile package is technical, not
  scientific: SVG figures still need conversion to PDF/PNG or regeneration from
  existing evidence tables.
- Next active step:
  - `configs/experiments/arxiv_figure_conversion_compile_plan.yaml`
- The arXiv figure conversion step is now complete:
  - five PDF figures were regenerated from existing evidence tables.
  - venue `main.tex` has no missing referenced PDF figures.
  - report: `docs/SAS_CERT_ARXIV_COMPILE_REPORT.md`
- Remaining blocker is local LaTeX tooling, not evidence or figure assets:
  - `latexmk`, `pdflatex`, `xelatex`, `lualatex`, `tectonic`, and `chktex` are
    unavailable in the current environment.
- Next active step:
  - `configs/experiments/arxiv_latex_tooling_or_external_compile_plan.yaml`
- Local LaTeX tooling has now been installed and the arXiv-first package
  compiles:
  - output: `paper/sas_cert_diagnostic_certificate_venue/main.pdf`
  - pages: `11`
  - final log: 0 LaTeX warnings, 0 overfull boxes, 0 undefined references, 0
    fatal errors.
  - report: `docs/SAS_CERT_ARXIV_FINAL_COMPILE_REPORT.md`
- Next active step:
  - `configs/experiments/arxiv_submission_bundle_review_plan.yaml`
- Remaining paper-engineering work is now submission bundle hygiene and metadata,
  not evidence generation or compilation.

## Anti-Scope-Creep Rules

- Do not continue MIRepNet mainline work.
- Do not continue paper reproduction unless it directly blocks the SAS-Cert question.
- Do not add EEGPT until ST/CBraMod PhysioNetMI is complete.
- Do not add emotion datasets until the two-backbone one-dataset test is resolved.
- Do not add hard Top50 as a main method.
- Do not use target test data for ranking, thresholding, score normalization, checkpoint selection, or seed selection.
