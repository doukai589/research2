# SAS-Cert Figure and Table Polish Report

## Verdict

```text
figure_table_assets_polished_as_draft_quality_without_data_changes
```

The generated figure and table assets are internally consistent with the
manuscript plan and were produced from existing CSV/JSON evidence only.

## Assets

Figures:

- `figure1_certificate_overview.svg`: conceptual certificate overview.
- `figure2_diagnostic_auc.svg`: scalar score failure and component recovery.
- `figure3_component_specificity_heatmap.svg`: component AUC heatmap by bad type.
- `figure4_training_policy_non_promotion.svg`: training-policy deltas and non-promotion.
- `figure5_causal_chain.svg`: diagnostic-to-training causal chain.

Tables:

- `table1_protocol_and_claim_boundary.md`
- `table2_diagnostic_auc_summary.md`
- `table3_bad_type_component_auc.md`
- `table4_training_policy_non_promotion.md`
- `table5_claim_support.md`
- `appendix_protocol_leakage_audit.md`

## Data Preservation

- No values were changed.
- No new experiments were run.
- No new claims were introduced.
- Figures were generated from the existing diagnostic pack summary tables.

## Remaining Layout Work

| Item | Status | Action |
|---|---|---|
| Figure 1 | draft-ready | Convert to final vector style after venue template is known. |
| Figure 2 | draft-ready | Consider adding confidence intervals only if already available; do not invent uncertainty bars. |
| Figure 3 | appendix-preferred | Too dense for main text; use compact summary in main text and keep full heatmap/table in appendix. |
| Figure 4 | draft-ready | Keep non-promotion labels visible; avoid leaderboard framing. |
| Figure 5 | draft-ready | Suitable as conceptual discussion figure. |
| Table 3 | appendix-preferred | Full bad-type component AUC table is long. |

## Claim Safety

The figure/table package preserves the manuscript boundary:

```text
diagnostic certificate supported; deployable weighting/rejection policy not promoted
```
