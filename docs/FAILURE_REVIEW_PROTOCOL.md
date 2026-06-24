# Failure Review Protocol

This protocol adapts the local `复盘助手` prompt into a lightweight EEG experiment process.

Use it when an experiment is executed correctly but fails its Go/No-Go criteria.

## Principle

Do not immediately abandon the hypothesis. First identify which hidden premise failed.

Also do not audit forever. A failed run gets one focused review cycle before deciding whether to rerun, revise, or park.

## Review Steps

### 1. Facts Only

Record quantitative facts:

- Metrics for each group
- Deltas versus primary baseline
- Subject/seed win rates
- Worst-subject result
- Calibration metrics: ECE, NLL, Brier
- Any NaN/OOM/runtime deviations

Do not explain yet.

### 2. Expected vs Actual

Make a table:

| Expectation | Actual Result | Verdict | Key Data |
|---|---|---|---|

Use `supported`, `rejected`, or `uncertain`.

### 3. Causal Chain

Write the causal chain:

```text
Bad/clean augmentation separable
→ cert score ranks useful samples higher
→ soft weighting/rejection changes training signal
→ few-shot target adaptation improves
→ calibration does not degrade
```

Mark each link:

- `supported`
- `rejected`
- `missing data`

The first broken link is the main failure point.

### 4. Two or Three Explanations

For the broken link, propose at most three explanations.

Each explanation must include:

- what it explains
- what it does not explain
- what single extra measurement would distinguish it

### 5. One Next Experiment

Choose one focused next experiment.

Rules:

- It must distinguish the top explanations.
- It should reuse existing outputs/configs when possible.
- It should not expand to a new dataset/model unless the failure point requires it.

### 6. Decision

Choose exactly one:

- `rerun_same_protocol`: execution/data issue likely
- `revise_cert`: score direction/component issue likely
- `revise_training`: weighting/loss/fine-tuning issue likely
- `park_branch`: result not promising enough for current mainline
- `expand_scope`: only allowed after the above are insufficient

## Stop Rules

Avoid infinite auditing:

- No more than one diagnostic rerun per failure without a new hypothesis.
- No more than three new diagnostic files/scripts per failure cycle.
- If the same broken link persists twice, revise the method or park the branch.

## Output Template

Every failed durable run should produce:

```text
FAILURE_REVIEW.md
failure_review_summary.json
```

Place them in the run folder or workbench trial folder.

