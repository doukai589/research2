# SAS-Cert Diagnostic Certificate Draft Review

## Review Scope

Reviewed draft:

```text
docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT.md
```

Supporting evidence:

```text
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv
docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md
outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/compact_result.json
```

Constraints:

- No new experiments.
- No promotion of weighting/rejection policies.
- Every major numerical claim should be traceable to a file path.

## Review Answers

| Question | Verdict | Notes |
|---|---|---|
| Does the draft overclaim training utility? | passed | It explicitly says current weighting/rejection policies are not promoted and not supported as deployable methods. |
| Are all numeric claims traceable to files? | passed | The draft points to `claim_support_table.csv`, compact result, diagnostic AUC table, training policy table, and protocol audit table. |
| Is the diagnostic contribution clear enough without promising accuracy gains? | passed | The draft frames SAS-Cert as a diagnostic certificate and separates diagnostic AUC from training utility. |
| What minimal text revisions are needed? | completed | AUC claims were tightened to the current synthetic mixed bad-augmentation diagnostic pool. |

## Revision Applied

The main wording risk was that phrases like "clean-vs-bad separation" could be
read as a broad real-world generalization. The evidence is narrower: it applies
to the current synthetic mixed bad-augmentation diagnostic pool on PhysioNetMI.

Revised wording in:

- Abstract
- Diagnostic framework
- Scalar score direction failure result
- Component diagnostics result
- Conclusion

## Remaining Gaps

These are writing gaps, not reasons to run new experiments now:

| Gap | Type | Next Action |
|---|---|---|
| Related work citations are not yet inserted into the draft | manuscript polish | Add citations from the existing reference library |
| Formal method equations are still prose-level | manuscript polish | Convert certificate axes into equations or pseudocode |
| Figure captions are not drafted | manuscript polish | Add one causal-chain figure and one evidence table figure |
| External dataset validation is absent | future work | Do not add until diagnostic paper plan is coherent |

## Decision

```text
draft_review_passed_with_minor_scope_revision
```

The draft is ready for the next writing step: manuscript polishing and citation
insertion. It should not trigger new experiments yet.

## Next Action

```text
prepare_manuscript_polish_and_citation_plan
```
