# Paper Reproduction Gate Report

This report keeps `code_exact`, `paper_exact`, and `executable_hybrid` separate. Hybrid outputs are not treated as exact reproduction.

## CBraMod Results

### code_exact
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| PhysioNet-MI | official_code_default | not_run |  |  | not_run | not_run_compute_guard |
| BCIC-IV-2a | official_code_default | not_run |  |  | not_run | not_run_compute_guard |

### paper_exact
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| PhysioNet-MI | 4-class | balanced_accuracy | 0.6417±0.0091 |  | failed | needs_paper_preprocessing_lmdb |
| PhysioNet-MI | 4-class | cohen_kappa | 0.5222±0.0169 |  | failed | needs_paper_preprocessing_lmdb |
| PhysioNet-MI | 4-class | weighted_f1 | 0.6427±0.01 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | E.9 4-class | balanced_accuracy | 0.5138±0.0066 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | E.9 4-class | cohen_kappa | 0.3518±0.0094 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a | E.9 4-class | weighted_f1 | 0.4984±0.0085 |  | failed | needs_paper_preprocessing_lmdb |
| BCIC-IV-2a LOSO key ingredients | LOSO key ingredients | balanced_accuracy | 0.7405±0.0635 |  | failed | not_run_missing_key_ingredients_implementation |
| BCIC-IV-2a LOSO key ingredients | LOSO key ingredients | auc_pr | 0.5997±0.0833 |  | failed | not_run_missing_key_ingredients_implementation |
| BCIC-IV-2a LOSO key ingredients | LOSO key ingredients | auroc | 0.7195±0.0682 |  | failed | not_run_missing_key_ingredients_implementation |

### executable_hybrid
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| PhysioNet-MI | forward_smoke_only | not_a_metric |  |  | not_scored | hybrid_forward_ready |
| BCIC-IV-2a | forward_smoke_only | not_a_metric |  |  | not_scored | hybrid_forward_ready |


## MIRepNet Results

### code_exact
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| BNCI2014001 | official_code_default | not_run |  |  | failed | official_repo_behavior_unclear |
| BNCI2014001-4 | official_code_default | not_run |  |  | failed | official_repo_behavior_unclear |

### paper_exact
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| BNCI2014001 | likely 2-class left/right | accuracy_full_finetune | 0.8177±0.0027 |  | failed | adapter_needed |
| BNCI2014001 | likely 2-class left/right | accuracy_linear_probe | 0.7536±0.0226 |  | failed | adapter_needed |
| BNCI2014001-4 | 4-class full | accuracy_full_finetune | 0.6414±0.0031 |  | failed | adapter_needed |
| BNCI2014001-4 | 4-class full | accuracy_linear_probe | 0.5194±0.0184 |  | failed | adapter_needed |

### executable_hybrid
| Dataset | Task | Metric | Paper | Local | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| BNCI2014001 | adapter_hybrid | not_run |  |  | failed | adapter_needed |
| BNCI2014001-4 | adapter_hybrid | not_run |  |  | failed | adapter_needed |


## Paper-Code Conflict Table

| Model | Dataset | Conflict | Chosen Protocol | Expected Impact |
| --- | --- | --- | --- | --- |
| CBraMod | PhysioNet-MI | preprocessing_and_entrypoint | paper_exact blocked until exact LMDB is generated; code_exact kept separate | Large; split/preprocessing mismatch can dominate metrics. |
| CBraMod | BCIC-IV-2a | split_and_preprocessing | paper_exact blocked until exact LMDB is generated; hybrid forward smoke not scored | Large; exact paper comparison invalid without exact preprocessing. |
| CBraMod | BCIC-IV-2a LOSO | missing_method_components | not_run_missing_key_ingredients_implementation | Large; paper target is much higher and depends on ingredients. |
| MIRepNet | BNCI2014001 | dataset_code_unreleased | paper_exact blocked; code_exact unavailable | Blocking; any local adapter is hybrid/paper reconstruction, not code exact. |
| MIRepNet | BNCI2014001-4 | missing_processed_data_and_adapter | adapter audit before reproduction | Blocking until adapter is validated. |
| MIRepNet | BNCI2014001/BNCI2014001-4 | dependency_and_input_shape | record missing dependency; no global install | Blocking for code execution; high risk if bypassed. |

## Discrepancy Analysis

- CBraMod paper_exact: not run; likely reason is missing exact regenerated/audited LMDB preprocessing and full 50-epoch fine-tune.
- MIRepNet paper_exact: not run; likely reason is adapter/dependency and unreleased official BNCI2014001 pipeline.
- Executable hybrid: only forward/audit status exists; not a reproduction result.

## Decision

- overall_decision: `ADAPTER_NEEDED`
- safe_for_sas_cert_next_stage: `False`
- next_action: Build exact preprocessing/LMDB for CBraMod paper runs and validate MIRepNet 45-channel interpolation+EA adapter before claiming paper reproduction.

## Compact JSON

```json
{
  "cbramod": {
    "bcic_iv_2a_e9": {
      "local_metrics": {},
      "matched_or_close": null,
      "paper_metrics": {
        "balanced_accuracy": {
          "mean": 0.5138,
          "std": 0.0066
        },
        "cohen_kappa": {
          "mean": 0.3518,
          "std": 0.0094
        },
        "weighted_f1": {
          "mean": 0.4984,
          "std": 0.0085
        }
      },
      "ran": false,
      "status": "failed"
    },
    "bcic_iv_2a_loso_key_ingredients": {
      "ran": false,
      "reason": "failed"
    },
    "physionet_mi": {
      "local_metrics": {},
      "matched_or_close": null,
      "paper_metrics": {
        "balanced_accuracy": {
          "mean": 0.6417,
          "std": 0.0091
        },
        "cohen_kappa": {
          "mean": 0.5222,
          "std": 0.0169
        },
        "weighted_f1": {
          "mean": 0.6427,
          "std": 0.01
        }
      },
      "ran": false,
      "status": "failed"
    }
  },
  "mirepnet": {
    "adapter_ready": false,
    "bnci2014001": {
      "local_metrics": {},
      "matched_or_close": null,
      "paper_metrics": {
        "accuracy_full_finetune": {
          "mean": 0.8177,
          "std": 0.0027
        },
        "accuracy_linear_probe": {
          "mean": 0.7536,
          "std": 0.0226
        }
      },
      "ran": false,
      "status": "failed"
    },
    "bnci2014001_4": {
      "local_metrics": {},
      "matched_or_close": null,
      "paper_metrics": {
        "accuracy_full_finetune": {
          "mean": 0.6414,
          "std": 0.0031
        },
        "accuracy_linear_probe": {
          "mean": 0.5194,
          "std": 0.0184
        }
      },
      "ran": false,
      "status": "failed"
    }
  },
  "next_action": "Build exact preprocessing/LMDB for CBraMod paper runs and validate MIRepNet 45-channel interpolation+EA adapter before claiming paper reproduction.",
  "overall_decision": "ADAPTER_NEEDED",
  "safe_for_sas_cert_next_stage": false,
  "stage": "paper_reproduction_gate",
  "status": "partial",
  "warnings": [
    "CBraMod paper_exact did not run; only gate/failure audit is available.",
    "MIRepNet adapter not ready; MIRepNet paper_exact did not run."
  ]
}
```
