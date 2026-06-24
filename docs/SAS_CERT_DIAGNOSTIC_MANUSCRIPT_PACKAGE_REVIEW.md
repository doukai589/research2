# SAS-Cert Diagnostic Manuscript Package Review

## Review Scope

Reviewed package:

- `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md`
- `docs/SAS_CERT_CITATION_PLAN.md`
- `docs/SAS_CERT_FIGURE_TABLE_PLAN.md`
- `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`

Review config:

```text
configs/experiments/diagnostic_manuscript_review_next.yaml
```

Constraints:

- No new experiments.
- No new claims without traceable evidence.
- No promotion of current weighting/rejection policies as deployable methods.

## Executive Verdict

```text
package_review_passed_with_submission_preparation_gaps
```

The manuscript package is internally coherent and aligned with the current
evidence boundary. It is ready for bibliography extraction and figure/table
generation, but not yet ready as a clean submission draft because citation
placeholders still need BibTeX resolution and one citation-planning sentence
should be removed from the abstract body.

## Checklist

| Item | Verdict | Evidence |
|---|---|---|
| Citation placeholders are mapped | passed_with_minor_gap | `docs/SAS_CERT_CITATION_PLAN.md` maps all intended placeholders to local references or `选题.txt` / `参考论文集` anchors. Exact BibTeX keys still need extraction. |
| Figure/table plans have source paths | passed | All planned main and appendix tables/figures point to existing CSV/JSON/MD files. |
| Claims match claim-support table | passed | Supported claims C1-C8 appear with conservative wording; unsupported claims C9-C10 are not asserted. |
| Weighting/rejection policy is not promoted | passed | Draft repeatedly states current policies are not promoted and not deployable. |
| Protocol leakage boundary is preserved | passed | The figure plan and draft point to `protocol_leakage_audit.csv`; no target-heldout use is introduced in the writing plan. |
| No new experiments added | passed | This review only reads and organizes existing manuscript/evidence artifacts. |

## Citation Audit

Mapped citation placeholders:

| Placeholder | Status | Current source anchor |
|---|---|---|
| `[Wang2025CBraMod]` | mapped | `docs/references/Wang 等 - 2025 - CBraMod A Criss-Cross Brain Foundation Model for EEG Decoding.pdf` |
| `[Liu2026MIRepNet]` | mapped | `docs/references/Liu 等 - 2026 - MIRepNet A pipeline and pre-trained model for EEG-based motor imagery classification.pdf` |
| `[RE199]` | mapped_needs_bibtex | `选题.txt` foundation-model marginal-gain note |
| `[RE132]` | mapped_needs_bibtex | `选题.txt` LaBraM+LoRA low-label note |
| `[RE309]` | mapped_needs_bibtex | `选题.txt` and `参考论文集/论文集_004.txt` JSCCRA entry |
| `[RE334]` | mapped_needs_bibtex | `选题.txt` and `参考论文集/论文集_004.txt` factorized latent note |
| `[RE167]` | mapped_needs_bibtex | `选题.txt` MI frequency-time prior note |
| `[RE181]` | mapped_needs_bibtex | `选题.txt` covariance/RCC note |
| `[RE333]` | mapped_needs_bibtex | `参考论文集/论文集_004.txt` EEGANet entry |
| `[RE342]` | mapped_needs_bibtex | `参考论文集/论文集_004.txt` automatic ICA artifact classifier entry |
| `[RE185]` | mapped_needs_bibtex | `选题.txt` EEGTune note |
| `[RE332]` | mapped_needs_bibtex | `选题.txt` small augmentation gain note |

Minor gap:

- `ST-EEGFormer-small` is used as a locked backbone in the local experiments,
  but a formal paper/source citation is not yet recorded in the citation plan.
  The plan already says to cite it only after the repository/paper source is
  formally recorded.

Submission-preparation action:

- Extract exact BibTeX entries for the RE placeholders before converting the
  draft to a submission format.
- Remove or move the line beginning `Citation placeholders:` from the abstract
  body; it is useful for project tracking but should not remain inside a
  submission abstract.

## Figure and Table Traceability Audit

All planned source files exist:

| Planned item | Source path | Status |
|---|---|---|
| Figure 1 protocol overview | `protocol_leakage_audit.csv`, `failure_mode_definitions.csv` | exists |
| Figure 2 diagnostic AUC bars | `diagnostic_auc_summary.csv` | exists |
| Figure 3 component heatmap | `bad_type_component_auc.csv` | exists |
| Figure 4 training non-promotion | `training_policy_summary.csv`, ST locked confirm compact JSON | exists |
| Figure 5 causal chain | `docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md`, `claim_support_table.csv` | exists |
| Table 1 protocol and claim boundary | `protocol_leakage_audit.csv`, `claim_support_table.csv` | exists |
| Table 2 diagnostic AUC summary | `diagnostic_auc_summary.csv` | exists |
| Table 3 bad-type component AUC | `bad_type_component_auc.csv` | exists |
| Table 4 training policy non-promotion | `training_policy_summary.csv` | exists |
| Table 5 claim support | `claim_support_table.csv` | exists |
| Appendix protocol leakage | `protocol_leakage_audit.csv` | exists |
| Appendix number traceability | `number_traceability.csv` | exists |
| Appendix failure-mode definitions | `failure_mode_definitions.csv` | exists |

## Claim Safety Audit

| Claim area | Verdict | Notes |
|---|---|---|
| SAS-Cert as diagnostic certificate | supported | Matches C1. |
| Old scalar SAS direction failure | supported | Matches C2. |
| Artifact-gate physio strongest diagnostic variant | supported | Matches C3. |
| Current weighting/rejection not deployable | supported | Matches C4, C9, C10. |
| CBraMod repaired classification gain with calibration risk | supported | Matches C5. |
| ST SoftWeight positive mean but unreliable | supported | Matches C6. |
| Candidate-only utility alignment weak | supported | Matches C7. |
| No detected protocol leakage in audited outputs | supported | Matches C8. |

No prohibited claim was found. In particular, the polished draft does not claim
that SAS-Cert reliably improves few-shot accuracy across subjects/seeds or that
it is ready as an augmentation-selection method.

## Minimal Submission Gaps

These are writing/package gaps, not reasons to run experiments:

| Gap | Priority | Required action |
|---|---:|---|
| Placeholder citations are not BibTeX-resolved | high | Create a bibliography file and replace project placeholders with stable keys. |
| Abstract contains a project-management citation-placeholder sentence | high | Move it to the citation plan or remove it from submission draft. |
| Figures and tables are planned but not rendered | high | Generate manuscript-ready tables and simple figures from existing CSV/JSON only. |
| ST-EEGFormer-small formal citation/source is not recorded | medium | Add a citation/source record before final submission. |
| Equations are plaintext rather than LaTeX | medium | Convert formulas to LaTeX notation in the submission draft. |
| Target venue style is undecided | low | Decide later after evidence package becomes camera-ready enough. |

## Decision

```text
proceed_to_bibliography_and_figure_generation
```

Next step should still avoid new experiments. The highest-value action is to
turn the manuscript package into a clean submission-preparation package:

1. Create BibTeX/citation trace artifacts from the existing references.
2. Generate figure/table drafts from existing CSV/JSON files.
3. Produce a submission-clean draft with project-management notes removed from
the abstract.
