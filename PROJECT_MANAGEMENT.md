# Project Management Log

This file is the long-running project ledger for `/ai/224duibishiyan/615新研究`.
Update it after each working conversation with what changed, what ran, where outputs were written, and what the current decision is.

## Current Status

- Workspace: `/ai/224duibishiyan/615新研究`
- Repository state: git repository initialized locally for code/config/document tracking; large outputs, weights, archives, and raw EEG data are excluded by `.gitignore`.
- Current research route: general EEG foundation model + cross-task SAS-Cert validation.
- Main active tracks:
  - `sas_cert_cbramod_mve`: SAS-Cert + frozen CBraMod augmentation-quality experiments on BCIC-IV-2a.
  - `third_party/backbones/STEEGFormer`: ST-EEGFormer code and official small/base/large release weights for next general-FM smoke.
  - `third_party/backbones/LaBraM`: LaBraM code and checkpoints for second general-FM readiness checks.
  - `third_party/backbones/EEGPT`: BINE022 EEGPT code plus user-supplied `eegpt_mcae_58chs_4s_large4E.ckpt`.
  - `outputs/paper_code_runs`: CBraMod paper-code reproduction on existing processed LMDBs, now archival/reference.
  - `outputs/mirepnet_full_paper_code`: MIRepNet reproduction attempts and BNCI2014001-4 adapter audits, now a parked side branch.
  - `outputs/setup_audit` and `outputs/setup_audit_step1`: backbone and dataset inventory / compatibility checks.

## Standing Rules

- Do not copy raw EEG datasets into project experiment folders.
- Keep processed adapters and generated features auditable with manifests.
- Record every substantial run here with command, protocol label, output paths, and result.
- After each answered user request, update this ledger with the decision,
  context learned, actions taken, and explicit no-op status when no research
  files, experiments, or outputs were changed.
- Mark hybrid / diagnostic runs clearly; do not mix them with exact paper reproduction.
- Preserve existing outputs unless explicitly asked to delete or archive them.
- New ideas start in `workbench/` before promotion to `sas_core/` or stable `scripts/`.
- Treat `third_party/` as external code; prefer wrappers/runtime patches and record them in `PATCHES.md`.
- For follow-up prompts, prefer iterating on the existing relevant workbench
  runner/config/report code instead of creating another nested trial or new
  duplicate file. Add new files only when they are required or clearly improve
  long-term maintainability.

## Running Summary

### Long-Term Research Scope

- Scientific question: determine whether an augmented EEG sample is a beneficial subject-style variation or a harmful task-content / physiology / artifact distortion for few-shot cross-subject adaptation of EEG foundation models.
- Locked current main scope:
  - Backbone 1: `CBraMod`
  - Backbone 2: `ST-EEGFormer-small`
  - Core dataset: `PhysioNetMI`, left/right MI, runs `R04/R08/R12`
  - Method policy:
    - CBraMod anchor branch: `SASCert_SoftAR_LS010`
    - ST current branch: subject heterogeneity diagnostic after `SoftWeight_noReject_LS010` and `SASCert_SoftAR_LS010` both failed reliability gates
- Secondary baseline: `LaBraM-base`
- Paused: `EEGPT-large4E`, `MIRepNet`, and non-blocking paper reproduction.
- Roadmap: `docs/RESEARCH_ROADMAP.md`
- Failure review protocol: `docs/FAILURE_REVIEW_PROTOCOL.md`
- Mainline config: `configs/experiments/mainline_scope.yaml`

### Current Main Route

- Main scientific question: whether SAS-Cert can certify and weight EEG augmentation samples reliably for few-shot / low-label adaptation across general EEG foundation models and tasks.
- Method policy is now backbone-aware:
  - CBraMod remains the historical anchor for `SASCert_SoftAR_LS010`.
  - ST-EEGFormer-small should not promote any method yet; both `SoftWeight_noReject_LS010` and `SASCert_SoftAR_LS010` improved mean metrics but failed subject/seed reliability.
  - ST support-routing development also failed LOSO validation, so the ST branch is paused at a clear diagnostic endpoint.
  - Active next step is CBraMod + PhysioNetMI matched validation.
- Main backbones: frozen/controlled CBraMod and ST-EEGFormer-small.
- Secondary backbone candidate: LaBraM, readiness/mini-eval only at first.
- Later candidate: EEGPT, smoke only after CBraMod/LaBraM path is stable.
- Dataset direction:
  - Keep BCIC-IV-2a as completed historical MI anchor.
  - Keep PhysioNetMI as the current active one-dataset test bed.
  - Do not add emotion datasets until ST subject heterogeneity and CBraMod PhysioNetMI matched validation are resolved.
- Stop spending mainline effort on MIRepNet reproduction, MIRepNet adapter repair, paper-exact reproduction gate, hard Top50, and MI-only framing.

### 2026-06-22 CBraMod PhysioNetMI Matched Mini Matrix

- Trial directory: `workbench/20260622_cbramod_physionetmi_sascert_matched`
- Purpose: run the CBraMod anchor on the same PhysioNetMI few-shot SAS-Cert protocol used for ST-EEGFormer-small.
- New runner:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/03_run_cbramod_mini_matrix.py`
- Inputs:
  - Shared PhysioNetMI cache: `outputs/foundation_physio_mi_fullfinetune/data/physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz`
  - Frozen CBraMod feature cache: `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_original_features.npz`
- Command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260622_cbramod_physionetmi_sascert_matched/03_run_cbramod_mini_matrix.py --device cuda --targets 90 91 92 --seeds 20 21 --source-epochs 30 --finetune-epochs 80 --batch-size 64 --feature-batch-size 32 --output-tag mini`
- Scope:
  - Targets: `90,91,92`
  - Seeds: `20,21`
  - Groups: `NaiveAug_LS010`, `ArtifactReject_LS010`, `SoftWeight_noReject_LS010`, `SASCert_SoftAR_LS010`
  - Backbone: frozen CBraMod only; classifier head training only.
  - Raw data copied: `false`
  - Raw augmented arrays saved: `false`
  - Target test used for ranking/threshold/training: `false`
- Mini result:
  - NaiveAug: BAcc `0.5349`, Macro-F1 `0.4983`, ECE `0.0843`, NLL `0.6931`.
  - ArtifactReject: BAcc `0.5302`, Macro-F1 `0.5001`, ECE `0.0923`, NLL `0.6931`.
  - SoftWeight no reject: BAcc `0.5360`, Macro-F1 `0.4925`, ECE `0.0866`, NLL `0.6937`.
  - SASCert SoftAR: BAcc `0.5473`, Macro-F1 `0.4988`, ECE `0.1120`, NLL `0.6932`.
  - SASCert SoftAR vs Naive: BAcc `+1.24pp`, Macro-F1 `+0.05pp`, ECE `+2.77pp`, NLL `+0.00018`.
  - Subject win rate by Macro-F1: `0.3333`; seed win rate by Macro-F1: `0.5000`.
- Decision:
  - `do_not_expand_to_full_yet`
  - This is diagnostic, not promoted. CBraMod SoftAR gives a small BAcc lift but does not materially improve Macro-F1 and worsens calibration beyond the gate.
  - Next step should be a bounded failure review of the CBraMod PhysioNetMI feature/training link, not a new augmentation variant or full matrix.
- Primary outputs:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_mini.md`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_mini.json`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_physionetmi_sascert_metrics_mini.csv`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_physionetmi_paired_comparison_mini.csv`

### 2026-06-22 CBraMod PhysioNetMI Failure Review And Repaired Score Mini

- Trial directory: `workbench/20260622_cbramod_physionetmi_sascert_matched`
- Failure review script:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/04_failure_review_cbramod_physionetmi.py`
- Runner update:
  - `03_run_cbramod_mini_matrix.py` now supports `--experiment repaired`.
  - Default `--experiment current` preserves the original mini matrix behavior.
- Failure review facts:
  - NoAug diagnostic: BAcc `0.5424`, Macro-F1 `0.5159`, ECE `0.0817`.
  - Source-only target diagnostic: BAcc `0.5068`, Macro-F1 `0.5001`, ECE `0.0043`.
  - Current SAS total clean-vs-bad AUC: `0.1969`.
  - Direction-fixed total score AUC: `0.8911`.
  - Artifact-gate physio score AUC: `0.9022`.
  - Direction flags: content and artifact_safe are inverted under the mixed bad candidate pool; physio and style are not inverted.
- Failure review decision:
  - `revise_cert_calibration`
  - First broken link: current mixed-bad SAS score direction is wrong on PhysioNetMI.
  - Interpretation: the useful certificate signal exists, but the current total score combines components in the wrong direction for this backbone/dataset.
- Repaired mini command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260622_cbramod_physionetmi_sascert_matched/03_run_cbramod_mini_matrix.py --device cuda --targets 90 91 92 --seeds 20 21 --source-epochs 30 --finetune-epochs 80 --batch-size 64 --feature-batch-size 32 --output-tag repaired_mini --experiment repaired`
- Repaired score:
  - `score_artifact_gate_physio = physio_score`, with artifact-risk p90 samples set to `0` before rank weighting.
- Repaired mini result:
  - NoAug: BAcc `0.5231`, Macro-F1 `0.4954`, ECE `0.0595`, NLL `0.6927`.
  - NaiveAug: BAcc `0.5349`, Macro-F1 `0.4983`, ECE `0.0843`, NLL `0.6931`.
  - RepairedSoftWeight: BAcc `0.5461`, Macro-F1 `0.5409`, ECE `0.1070`, NLL `0.6916`.
  - RepairedSoftWeight vs Naive: BAcc `+1.11pp`, Macro-F1 `+4.26pp`, ECE `+2.27pp`, NLL `-0.00148`, Brier `-0.00147`.
  - Subject win rate by Macro-F1: `0.6667`; seed win rate by Macro-F1: `1.0000`.
- Decision:
  - `classification_signal_repaired_but_calibration_gate_failed`
  - Do not expand to full CBraMod PhysioNetMI yet.
  - Next allowed step: define exactly one calibration-aware repaired score/loss and test it on the same mini scope.
  - Do not add new datasets, new backbones, or a broad method sweep.
- Primary outputs:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/FAILURE_REVIEW.md`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/failure_review_summary.json`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_repaired_mini.md`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_repaired_mini.json`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_physionetmi_sascert_metrics_repaired_mini.csv`

### 2026-06-22 CBraMod PhysioNetMI Calibration-Aware Repair Mini

- Trial directory: `workbench/20260622_cbramod_physionetmi_sascert_matched`
- Runner update:
  - `03_run_cbramod_mini_matrix.py` now supports `--experiment calibrated`.
  - The calibrated branch uses the same repaired artifact-gate physio soft weighting, then fits a scalar temperature using target support labels only.
  - Held-out target labels are still used only for final evaluation.
- Command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260622_cbramod_physionetmi_sascert_matched/03_run_cbramod_mini_matrix.py --device cuda --targets 90 91 92 --seeds 20 21 --source-epochs 30 --finetune-epochs 80 --batch-size 64 --feature-batch-size 32 --output-tag calibrated_mini --experiment calibrated`
- Scope:
  - Targets: `90,91,92`
  - Seeds: `20,21`
  - Groups: `NoAug_LS010`, `NaiveAug_LS010`, `RepairedSoftWeight_artifact_gate_physio_LS010`, `RepairedSoftWeightTemp_artifact_gate_physio_LS010`
- Temperature result:
  - Fitted temperature was `0.5` for all six folds.
  - This sharpened predictions rather than softening them.
- Calibrated mini result:
  - NoAug: BAcc `0.5231`, Macro-F1 `0.4954`, ECE `0.0595`, NLL `0.6927`.
  - NaiveAug: BAcc `0.5349`, Macro-F1 `0.4983`, ECE `0.0843`, NLL `0.6931`.
  - RepairedSoftWeight: BAcc `0.5461`, Macro-F1 `0.5409`, ECE `0.1070`, NLL `0.6916`.
  - RepairedSoftWeightTemp: BAcc `0.5461`, Macro-F1 `0.5409`, ECE `0.1064`, NLL `0.6940`.
  - RepairedSoftWeightTemp vs Naive: BAcc `+1.11pp`, Macro-F1 `+4.26pp`, ECE `+2.21pp`, NLL `+0.00094`, Brier `+0.00065`.
- Decision:
  - `simple_support_temperature_scaling_failed_calibration_gate`
  - `park_cbramod_physionetmi_full_expansion`
  - The repaired-score signal is scientifically useful, but it is not deployable as a successful method because the calibration gate still fails.
  - Next active step should be a cross-backbone certificate direction audit using existing ST/CBraMod score rows, or a return to ST reliability after that audit. Do not keep tuning CBraMod calibration on the same mini without a new hypothesis.
- Primary outputs:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/CBRAMOD_PHYSIONETMI_SASCERT_REPORT_calibrated_mini.md`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/compact_cbramod_physionetmi_sascert_result_calibrated_mini.json`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_physionetmi_sascert_metrics_calibrated_mini.csv`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_physionetmi_paired_comparison_calibrated_mini.csv`

### 2026-06-22 Cross-Backbone Certificate Direction Audit

- Trial directory: `workbench/20260622_cross_backbone_cert_direction_audit`
- Purpose: compare SAS-Cert component directions across the two locked backbones before adding another training variant.
- Script:
  - `workbench/20260622_cross_backbone_cert_direction_audit/01_cross_backbone_cert_direction_audit.py`
- Inputs:
  - ST source-tuned score rows: `workbench/20260622_steegformer_sascert_core/outputs/score_rows/st_source_tuned_full`
  - CBraMod frozen score rows: `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/score_rows/mini`
- Scope:
  - Dataset: `PhysioNetMI`
  - Targets: `90,91,92`
  - Seeds: `20,21`
  - Existing score rows only; no retraining, no new augmentation generation, no target held-out labels.
- Command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260622_cross_backbone_cert_direction_audit/01_cross_backbone_cert_direction_audit.py`
- Overall component AUC, high score means clean:
  - CBraMod:
    - content `0.3044`
    - style `0.6408`
    - physio `0.8444`
    - artifact_safe `0.3333`
    - current sas `0.1969`
    - artifact_gate_physio `0.9022`
  - ST-EEGFormer-small:
    - content `0.2722`
    - style `0.6408`
    - physio `0.8444`
    - artifact_safe `0.3333`
    - current sas `0.1662`
    - artifact_gate_physio `0.9022`
- Key bad-type result:
  - `content_score` on BadContent has a backbone direction conflict:
    - ST AUC `0.2630`
    - CBraMod AUC `0.9025`
  - `artifact_safe_score` is artifact-specific:
    - BadArtifact AUC `1.0000`
    - BadContent AUC `0.0000`
    - BadPhysio AUC `0.0000`
- Decision:
  - `revise_scalar_score_before_training_expansion`
  - Do not promote the fixed scalar `sas_score`.
  - Define a component-gated certificate rule before any new training expansion.
  - Scientific interpretation: SAS-Cert should be framed as a multi-dimensional reliability certificate, not a universal scalar ranker.
- Active next experiment in `configs/experiments/mainline_scope.yaml` is now:
  - `define_component_gated_certificate_rule`
- Primary outputs:
  - `workbench/20260622_cross_backbone_cert_direction_audit/TRIAL.md`
  - `workbench/20260622_cross_backbone_cert_direction_audit/outputs/CROSS_BACKBONE_CERT_DIRECTION_AUDIT.md`
  - `workbench/20260622_cross_backbone_cert_direction_audit/outputs/compact_cross_backbone_cert_direction_audit.json`
  - `workbench/20260622_cross_backbone_cert_direction_audit/outputs/component_auc_by_backbone.csv`
  - `workbench/20260622_cross_backbone_cert_direction_audit/outputs/component_auc_by_bad_type.csv`
  - `workbench/20260622_cross_backbone_cert_direction_audit/outputs/bad_type_direction_conflict_table.csv`
  - `configs/experiments/cross_backbone_cert_direction_audit.yaml`

### 2026-06-22 Component-Gated Certificate Rule v1

- Trial directory: `workbench/20260622_component_gated_cert_rule`
- Purpose: turn the cross-backbone direction audit into an explicit, auditable SAS-Cert rule before any new training expansion.
- Script:
  - `workbench/20260622_component_gated_cert_rule/01_define_and_validate_component_gated_rule.py`
- Rule:
  - `artifact_gate_pass = artifact_risk < fold_p90`
  - `base = 0.75 * physio_score + 0.25 * style_score`
  - `component_gated_v1 = ranknorm(base) * artifact_gate_pass`
  - `component_gated_v1_soft = ranknorm(base) * (0.2 + 0.8 * artifact_safe_rank)`
  - `content_score` is diagnostic-only, not a positive term.
- Reason:
  - `physio_score` and `style_score` were stable clean-high components.
  - `artifact_safe_score` works as an artifact gate but is inverted for non-artifact bad types if treated as a universal positive score.
  - `content_score` is backbone/bad-type dependent, especially on BadContent.
- Score-only result:
  - CBraMod:
    - current SAS AUC `0.1969`
    - artifact_gate_physio AUC `0.9022`
    - component_gated_v1 AUC `0.8395`
    - component_gated_v1_soft AUC `0.7383`
  - ST-EEGFormer-small:
    - current SAS AUC `0.1662`
    - artifact_gate_physio AUC `0.9022`
    - component_gated_v1 AUC `0.8395`
    - component_gated_v1_soft AUC `0.7383`
- Decision:
  - `component_gated_v1_defined_score_validated`
  - The interpretable v1 rule fixes the scalar-score direction failure, but `score_artifact_gate_physio` remains the strongest score-only baseline.
  - Next active step is a small ST reliability test comparing current ST soft weighting, `component_gated_v1`, and `score_artifact_gate_physio`.
- Active next experiment in `configs/experiments/mainline_scope.yaml` is now:
  - `small_st_reliability_test_component_gated_vs_artifact_gate_physio`
- Primary outputs:
  - `workbench/20260622_component_gated_cert_rule/TRIAL.md`
  - `workbench/20260622_component_gated_cert_rule/outputs/COMPONENT_GATED_CERT_RULE_V1.md`
  - `workbench/20260622_component_gated_cert_rule/outputs/compact_component_gated_cert_rule_v1.json`
  - `workbench/20260622_component_gated_cert_rule/outputs/component_gated_v1_component_auc.csv`
  - `workbench/20260622_component_gated_cert_rule/outputs/component_gated_v1_bad_type_auc.csv`
  - `workbench/20260622_component_gated_cert_rule/outputs/component_gated_v1_gate_stats.csv`
  - `configs/experiments/component_gated_cert_rule_v1.yaml`

### General EEG-FM Backbone Download Audit

- Output directory: `outputs/backbone_download_audit`.
- ST-EEGFormer:
  - Code path: `third_party/backbones/STEEGFormer`
  - Official GitHub release weights downloaded:
    - `pretrained_weights/ST-EEGFormer-small/checkpoint-300.pth`
    - `pretrained_weights/ST-EEGFormer-base/checkpoint-288.pth`
    - `pretrained_weights/ST-EEGFormer-large/large_weights_only_196.pth`
  - `largeV2` release recorded but intentionally not downloaded for first smoke.
  - Recommended first smoke: small, then base.
- LaBraM:
  - Code path: `third_party/backbones/LaBraM`
  - `checkpoints/labram-base.pth` exists and loads.
  - `checkpoints/vqnsp.pth` exists and loads.
  - Files are not Git LFS pointers.
- EEGPT:
  - Code path: `third_party/backbones/EEGPT`
  - Checkpoint path: `third_party/backbones/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
  - Source: user-supplied `/ai/224duibishiyan/615新研究/25866970.zip`; torch load succeeded.
  - Status: ready for local experiments, with runtime compatibility patches required in the PhysioNetMI runner.

### SAS-Cert-CBraMod

- V1 full MVE completed under `sas_cert_cbramod_mve/outputs`.
- V1 key result: Layer 2 overall AUC `0.6217`, below the `0.70` target.
- V1 strong signals:
  - BadArtifact AUC `0.9474`
  - BadContent AUC `0.8009`
  - BadPhysio AUC `0.1595`, indicating score direction or definition problem.
- V1 Layer 3 hard Top50 did not beat NaiveAug.
- V2 moved focus to SoftWeightArtifactReject.
- V3/V4 supported calibrated soft weighting as the stronger branch.
- V5 locked confirmatory run ended with decision `LABEL_SMOOTHING_CONFOUNDED`; treat label smoothing conclusions cautiously.

Primary reports:

- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_MVE_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V2_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V3_CALIBRATION_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V4_CONFIRMATORY_REPORT.md`
- `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V5_LOCKED_CONFIRMATORY_REPORT.md`

### CBraMod Paper-Code Reproduction

- Protocol label: `code_exact_on_existing_paper_aligned_lmdb`
- Output directory: `outputs/paper_code_runs`
- PhysioNet-MI was close to paper target.
- BCIC-IV-2a was substantially below paper target, suggesting BCIC preprocessing/split mismatch or generalization issue.

Primary report:

- `outputs/paper_code_runs/PAPER_CODE_RUN_REPORT.md`

### Foundation Backbones on PhysioNetMI

- Status: completed one shared full fine-tuning run for ST-EEGFormer-small, LaBraM-base, and EEGPT-large4E.
- Protocol label: `paper_aligned_common_protocol`.
- Output directory: `outputs/foundation_physio_mi_fullfinetune`.
- Dataset source: `/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files`; no raw EDF data copied.
- Task: PhysioNetMI left/right motor imagery, runs `R04/R08/R12`, labels `T1/T2`, 4-second windows.
- Split: subjects `1-70` train, `71-89` validation, `90-109` test.
- Preprocessing: 1-40 Hz bandpass, per-trial per-channel z-score, canonical cache `[4917, 64, 640]`.
- Command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps python3 scripts/40_run_physio_mi_foundation_fullfinetune.py --models st_eegformer_small labram_base eegpt_large4e --epochs 30 --batch-size 16 --eegpt-batch-size 4 --patience 10 --device cuda`
- Results:
  - ST-EEGFormer-small: test Acc `0.7667`, BAcc `0.7669`, macro-F1 `0.7635`, kappa `0.5336`, AUROC `0.8712`, ECE `0.1816`.
  - LaBraM-base: test Acc `0.6544`, BAcc `0.6547`, macro-F1 `0.6481`, kappa `0.3093`, AUROC `0.7122`, ECE `0.0620`.
  - EEGPT-large4E: test Acc `0.5211`, BAcc `0.5211`, macro-F1 `0.5209`, kappa `0.0421`, AUROC `0.5384`, ECE `0.4614`.
- Interpretation:
  - ST-EEGFormer-small is the strongest current PhysioNetMI candidate and clearly beats the earlier CBraMod PhysioNetMI reference around BAcc `0.6285`.
  - LaBraM-base is usable but weaker, modestly above the CBraMod reference under this shared protocol.
  - EEGPT-large4E is near chance and poorly calibrated in the current adapter/runtime; do not treat it as a strong mainline candidate without adapter/runtime repair.
- Compatibility notes:
  - Local dependencies installed only under `outputs/foundation_physio_mi_fullfinetune/local_python_deps`: `timm==0.9.16`, `einops==0.7.0`.
  - EEGPT needed runner-local patches for constrained-layer autocast and missing PyTorch-2 scaled-dot-product attention.
- Primary report:
  - `outputs/foundation_physio_mi_fullfinetune/PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md`

### ST-EEGFormer SAS-Cert Workbench

- Trial directory: `workbench/20260622_steegformer_sascert_core`
- Purpose: test whether the locked `SASCert_SoftAR_LS010` method transfers from the CBraMod anchor to ST-EEGFormer-small on PhysioNetMI.
- Reusable modules promoted into `sas_core`:
  - `sas_core/data/physionet_mi.py`
  - `sas_core/data/transforms.py`
  - `sas_core/backbones/steegformer.py`
  - `sas_core/metrics/classification.py`
  - `sas_core/utils/io.py`
  - `sas_core/utils/seed.py`
- Protocol:
  - Dataset: PhysioNetMI left/right MI, runs `R04/R08/R12`.
  - Source subjects: `1-70`.
  - Target subjects: `90-109`.
  - Seeds: `20,21,22,23,24`.
  - Support: target subject 5-shot per class.
  - Groups: `NaiveAug_LS010`, `ArtifactReject_LS010`, `SoftWeight_noReject_LS010`, `SASCert_SoftAR_LS010`.
- Full original-pretrained-feature run command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260622_steegformer_sascert_core/runner.py --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 --seeds 20 21 22 23 24 --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64`
- Full result:
  - NaiveAug: BAcc `0.5039`, Macro-F1 `0.4921`, ECE `0.2649`, NLL `0.8854`.
  - ArtifactReject: BAcc `0.5067`, Macro-F1 `0.4948`, ECE `0.2676`, NLL `0.8840`.
  - SoftWeight no reject: BAcc `0.5106`, Macro-F1 `0.4982`, ECE `0.2642`, NLL `0.8876`.
  - SASCert SoftAR: BAcc `0.5161`, Macro-F1 `0.5037`, ECE `0.2619`, NLL `0.8842`.
  - SASCert vs Naive: BAcc `+1.21pp`, Macro-F1 `+1.16pp`, ECE `-0.0030`, NLL `-0.0011`.
  - Subject win rate Macro-F1: `0.25`.
  - Seed win rate Macro-F1: `0.40`.
- Decision:
  - `revise_training`, not promote.
  - Mean SAS-Cert effect is positive, but subject/seed reliability failed and all groups are near chance.
  - First broken link is likely the original pretrained frozen ST feature space, not necessarily the SAS-Cert rule.
- Failure review:
  - `workbench/20260622_steegformer_sascert_core/FAILURE_REVIEW.md`
  - `workbench/20260622_steegformer_sascert_core/failure_review_summary.json`
- Runner update:
  - `runner.py` now supports `--steegformer-state-dict` and cache `--feature-tag`, so a source-tuned feature rerun will not reuse the original pretrained feature cache.
- Next focused run:
  - Use `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt` as a frozen source/validation-trained ST feature extractor.
  - Keep the same dataset, target subjects, seeds, augmentation pool, and SAS-Cert parameters.
- Source-tuned feature rerun:
  - State dict: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
  - Feature tag: `st_source_tuned_seed3407`
  - Output tag: `st_source_tuned_full`
  - Load audit: `loaded_keys=106`, no missing/unexpected keys.
  - NaiveAug: BAcc `0.7088`, Macro-F1 `0.7045`, ECE `0.2079`, NLL `0.6853`.
  - ArtifactReject: BAcc `0.7107`, Macro-F1 `0.7064`, ECE `0.2112`, NLL `0.6854`.
  - SoftWeight no reject: BAcc `0.7153`, Macro-F1 `0.7109`, ECE `0.2082`, NLL `0.6832`.
  - SASCert SoftAR: BAcc `0.7149`, Macro-F1 `0.7108`, ECE `0.2056`, NLL `0.6810`.
  - SASCert vs Naive: BAcc `+0.61pp`, Macro-F1 `+0.63pp`, ECE `-0.0023`, NLL `-0.0044`, Brier `-0.0062`.
  - Subject win rate Macro-F1: `0.25`.
  - Seed win rate Macro-F1: `0.00`.
- Updated decision after source-tuned rerun:
  - ST-EEGFormer-small remains in the mainline because source-tuned features produce useful target few-shot performance.
  - Locked `SASCert_SoftAR_LS010` is not promoted as a success on ST because reliability criteria still fail.
  - `SoftWeight_noReject_LS010` is the immediate ST-specific branch to inspect; artifact reject may be over-pruning useful target-support augmentations in this feature space.
  - Do not expand to new datasets yet. Next step should diagnose artifact gate versus soft weighting under the same source-tuned ST setup.
- Artifact-gate diagnostic under source-tuned ST:
  - Script: `workbench/20260622_steegformer_sascert_core/diagnose_artifact_gate.py`
  - Inputs: source-tuned score rows and paired metrics only; no retraining and no target-test ranking.
  - p90 gate:
    - reject rate `0.10`
    - clean reject rate `0.00`
    - BadArtifact reject rate `0.50`
    - decision `artifact_gate_precise_but_conservative`
  - p80 gate:
    - reject rate `0.20`
    - clean reject rate `0.00`
    - BadArtifact reject rate `1.00`
    - score-space decision looked cleaner than p90, so it was smoke-tested in training.
  - p70 gate:
    - reject rate `0.30`
    - clean reject rate `0.25`
    - decision `artifact_gate_overprunes_clean_or_useful_candidates`
  - p80 mini training check:
    - scope `targets=90,91,92`, `seeds=20,21`
    - `SASCert_SoftAR_LS010` p80 vs Naive: BAcc `-0.0080`, Macro-F1 `-0.0081`, ECE `-0.0141`, NLL `+0.0024`
    - p80 improved calibration but hurt classification, so do not run full p80 now.
  - Updated method-local decision:
    - For ST classification, prefer `SoftWeight_noReject_LS010` as the next branch.
    - Keep artifact gate as a calibration diagnostic or optional calibration-control branch, not as the current ST main method.
- SoftWeight no-reject paired summaries from existing full source-tuned outputs:
  - `SoftWeight_noReject_LS010` vs `NaiveAug_LS010`:
    - BAcc `+0.0065`
    - Macro-F1 `+0.0064`
    - ECE `+0.0003`
    - NLL `-0.0022`
    - Brier `-0.0057`
    - subject win rate Macro-F1 `0.15`
    - seed win rate Macro-F1 `0.00`
  - `SoftWeight_noReject_LS010` vs `SASCert_SoftAR_LS010`:
    - BAcc `+0.0003`
    - Macro-F1 `+0.0001`
    - ECE `+0.0026`
    - NLL `+0.0022`
    - subject win rate Macro-F1 `0.05`
    - seed win rate Macro-F1 `0.00`
  - Updated ST method decision:
    - `do_not_promote_any_st_method_yet`
    - Mean gains exist, but reliability gates fail for both SoftWeight and SoftAR.
    - Next config is `configs/experiments/steegformer_physionetmi_subject_heterogeneity.yaml`.
- Subject heterogeneity diagnostic:
  - Script: `workbench/20260622_steegformer_sascert_core/diagnose_subject_heterogeneity.py`
  - Report: `workbench/20260622_steegformer_sascert_core/outputs/subject_heterogeneity/SUBJECT_HETEROGENEITY_REPORT.md`
  - Primary comparison: `SoftWeight_noReject_LS010` vs `NaiveAug_LS010`
  - Secondary comparison: `SASCert_SoftAR_LS010` vs `NaiveAug_LS010`
  - Primary subject win rate: `0.60`
  - Secondary subject win rate: `0.65`
  - Primary mean subject delta Macro-F1: `+0.0064`
  - Secondary mean subject delta Macro-F1: `+0.0063`
  - Primary losing subjects: `92,93,97,101,102,105,107,109`
  - Primary winning subjects: `90,91,94,95,96,98,99,100,103,104,106,108`
  - Correlation between baseline Macro-F1 and SoftWeight delta Macro-F1: `-0.4209`
  - Interpretation:
    - Soft weighting tends to help lower-baseline target subjects and can hurt or fail to help strong-baseline subjects.
    - This explains positive mean gains with weak universal reliability.
  - Updated ST next step:
    - Do not promote any ST method yet.
    - Develop a support-only routing rule on validation subjects `71-89`, then freeze it before final target subjects `90-109`.
    - New active config: `configs/experiments/steegformer_physionetmi_support_routing_dev.yaml`.
- Support-routing dev on validation subjects:
  - Output tag: `st_source_tuned_routing_dev`
  - Subjects: `71-89`
  - Folds: `95`
  - Report: `workbench/20260622_steegformer_sascert_core/outputs/support_routing_dev/SUPPORT_ROUTING_DEV_REPORT.md`
  - Constant strategies:
    - NaiveAug Macro-F1 `0.6621`, BAcc `0.6665`, ECE `0.2350`, NLL `0.8119`
    - SoftWeight no reject Macro-F1 `0.6649`, BAcc `0.6700`, ECE `0.2316`, NLL `0.8091`
    - SASCert SoftAR Macro-F1 `0.6654`, BAcc `0.6700`, ECE `0.2333`, NLL `0.8062`
  - Full-dev best threshold rule:
    - `mean_artifact_risk<=475.104 ? SASCert_SoftAR_LS010 : NaiveAug_LS010`
    - Macro-F1 `0.6675`
  - Leave-one-subject-out routing:
    - Macro-F1 `0.6594`, BAcc `0.6641`, ECE `0.2327`, NLL `0.8132`
  - Decision:
    - `do_not_freeze_routing_rule`
    - The apparent full-dev routing gain does not survive subject-held-out validation.
    - Do not apply routed ST methods to final subjects.
  - Updated mainline:
    - ST branch is paused at a clear diagnostic endpoint.
    - Active next config is now `configs/experiments/cbramod_physionetmi_sascert_matched.yaml`.

### CBraMod PhysioNetMI Matched Validation

- Trial directory: `workbench/20260622_cbramod_physionetmi_sascert_matched`
- Stable config: `configs/experiments/cbramod_physionetmi_sascert_matched.yaml`
- Status: `feature_cache_completed`
- Purpose:
  - Run the CBraMod anchor on the same PhysioNetMI few-shot SAS-Cert protocol used for ST-EEGFormer-small.
  - Avoid cross-dataset comparison between CBraMod on BCIC-IV-2a and ST on PhysioNetMI.
- First required step:
  - Resolve local CBraMod code and checkpoint.
  - Reuse the existing PhysioNetMI loader/cache without copying raw data.
  - Confirm frozen CBraMod forward on one PhysioNetMI trial.
  - Confirm pooled feature shape and no NaN/Inf.
- Expected smoke outputs:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_smoke_report.json`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_smoke_log.txt`
- Smoke result:
  - Script: `workbench/20260622_cbramod_physionetmi_sascert_matched/01_smoke_cbramod_physionetmi.py`
  - PhysioNetMI cache reused: `outputs/foundation_physio_mi_fullfinetune/data/physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz`
  - Cache shape: `[4917,64,640]`
  - CBraMod input shape: `[8,64,4,200]`
  - Representation shape: `[8,64,4,200]`
  - Pooled feature shape: `[8,200]`
  - Checkpoint loaded keys: `211/211`
  - Missing/unexpected/shape mismatch: `0/0/0`
  - Feature NaN/Inf: `0/0`
  - Raw data copied: `false`
  - Resample policy: canonical PhysioNetMI 640 samples -> CBraMod 800 samples -> `[64,4,200]`
- Next step:
  - Run a small target/seed mini matrix.
- Feature cache result:
  - Script: `workbench/20260622_cbramod_physionetmi_sascert_matched/02_cache_cbramod_features.py`
  - Cache: `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_original_features.npz`
  - Manifest: `workbench/20260622_cbramod_physionetmi_sascert_matched/outputs/cbramod_original_features_manifest.json`
  - Feature shape: `[4917,200]`
  - Labels shape: `[4917]`
  - Subjects range: `1-109`
  - Runs: `4,8,12`
  - Feature NaN/Inf: `0/0`
  - Raw trial array saved: `false`

### MIRepNet Reproduction

- Status: parked side branch; no longer part of the main SAS-Cert route.
- Initial full run used a locally generated raw `.mat` T+E hybrid adapter.
- Result: mean accuracy `58.38%` on `BNCI2014001-4`, below paper target `64.14%`.
- GitHub issue audit indicated BNCI2014001 / BNCI2014001-4 should use MOABB first/session-T only.
- Rebuilt adapter with MOABB `BNCI2014_001`, `0train` only:
  - Shape `[2592, 22, 1000]`
  - 9 subjects x 288 trials
  - 4 classes x 648 trials
- Reran MIRepNet:
  - batch32: `57.76%`
  - batch8 default-like: `60.23%`
- Best corrected rerun improved over raw/T+E hybrid by `+1.85pp`, but remained `-3.91pp` below the paper `64.14%`.
- Interpretation: session/protocol correction helped, but does not fully explain the reproduction gap.

Primary report:

- `outputs/mirepnet_full_paper_code/MIRepNet_MOABB_SESSION_T_RERUN_REPORT.md`

## Conversation Log

### 2026-06-22

- User asked to organize the project and create a long-term project-management file.
- Added non-destructive management/index files instead of moving experiment outputs:
  - `PROJECT_MANAGEMENT.md`
  - `PROJECT_INDEX.md`
  - `scripts/README.md`
  - `outputs/README.md`
- Reason: existing scripts and reports reference current paths, so reorganizing by moving files would risk breaking reproducibility.
- User provided a new route decision: demote MIRepNet to an attempted-but-parked branch and move the main line to general EEG foundation models plus cross-task validation.
- Updated current route to `SAS_CERT_GENERAL_FM_CROSS_TASK_CORE_EXPERIMENTS`.
- Locked main method for future core experiments as `SASCert_SoftAR_LS010`; do not continue hard Top50 or paper-exact reproduction work as mainline.
- User chose to keep GPU memory reserved because the machine has an 80GB A800.
- Started `显存占位/start_gpu_hold.sh` with defaults:
  - tmux session: `gpu_hold`
  - GPU: `0`
  - reserved memory: `30GB`
  - log: `显存占位/hold_gpu_mem.log`
- Stop command when full GPU memory is needed: `./显存占位/stop_gpu_hold.sh`
- Downloaded and audited general EEG-FM backbone candidates:
  - ST-EEGFormer code + small/base/large official release weights.
  - LaBraM code + existing `labram-base.pth` / `vqnsp.pth` checkpoints.
  - EEGPT code; official checkpoint pending manual Figshare download.
- Generated:
  - `outputs/backbone_download_audit/backbone_download_inventory.csv`
  - `outputs/backbone_download_audit/backbone_download_inventory.json`
  - `outputs/backbone_download_audit/DOWNLOAD_BACKBONE_REPORT.md`
  - `outputs/backbone_download_audit/compact_backbone_download_result.json`
- User supplied EEGPT checkpoint zip `/ai/224duibishiyan/615新研究/25866970.zip`; extracted checkpoint to `third_party/backbones/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`.
- Implemented shared PhysioNetMI full fine-tuning runner:
  - `scripts/40_run_physio_mi_foundation_fullfinetune.py`
  - output root `outputs/foundation_physio_mi_fullfinetune`
  - protocol `paper_aligned_common_protocol`
- Ran full fine-tuning for:
  - `st_eegformer_small`
  - `labram_base`
  - `eegpt_large4e`
- Key result:
  - ST-EEGFormer-small is strongest on PhysioNetMI test (`BAcc=0.7669`, `Macro-F1=0.7635`).
  - LaBraM-base is moderate (`BAcc=0.6547`, `Macro-F1=0.6481`).
  - EEGPT-large4E is near chance and poorly calibrated (`BAcc=0.5211`, `ECE=0.4614`).
- Main output files:
  - `outputs/foundation_physio_mi_fullfinetune/PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md`
  - `outputs/foundation_physio_mi_fullfinetune/results/physio_mi_fullfinetune_metrics.csv`
  - `outputs/foundation_physio_mi_fullfinetune/results/compact_physio_mi_fullfinetune_result.json`
- Consolidated the PhysioNetMI comparison with the earlier CBraMod paper-code result for interpretation:
  - CBraMod official paper-code reference: `BAcc=0.6285`.
  - ST-EEGFormer-small shared full fine-tune: `BAcc=0.7669`.
  - LaBraM-base shared full fine-tune: `BAcc=0.6547`.
  - EEGPT-large4E shared full fine-tune: `BAcc=0.5211`.
- Route interpretation after reviewing the proposed plan:
  - Promote ST-EEGFormer-small into the main line as the first companion backbone beyond CBraMod.
  - Keep LaBraM-base as a secondary reliability/calibration baseline.
  - Pause EEGPT until adapter/runtime repair is explicitly needed.
  - Keep MIRepNet parked outside the main route.
  - Next most valuable experiment is ST-EEGFormer-small + PhysioNetMI SAS-Cert core validation, not more paper reproduction.
- Refactored workspace management structure to support long-term iteration:
  - Added `README.md`, `docs/PROJECT_WORKFLOW.md`, `RUN_REGISTRY.md`, `PATCHES.md`.
  - Added `workbench/` as the trial/intermediate box area.
  - Added `archive/` for failed/parked branches.
  - Added `sas_core/` package skeleton for reusable modules.
  - Added `configs/experiments/` with ST full fine-tune reference and ST SAS-Cert planned configs.
  - Added `outputs/runs/` as the normalized future run-output location.
  - Added `scripts/manage_trial.py` for creating isolated workbench trial boxes.
  - Added `scripts/SCRIPT_REGISTRY.md` to classify active, setup, and parked scripts.
  - Moved top-level zip archives into `artifacts/source_archives/`.
  - Moved reference papers into `docs/references/`.
  - Initialized local git repository and added `.gitignore` to exclude outputs, weights, raw EEG data, archives, and runtime logs.
- New trial workflow:
  - Create trial: `python scripts/manage_trial.py new <trial_name>`
  - Develop inside `workbench/<date>_<trial_name>/`
  - Promote reusable parts into `sas_core/` and stable configs only after success.
- Locked long-term research scope based on `选题.txt` and `复盘助手`:
  - Two main backbones: `CBraMod` and `ST-EEGFormer-small`.
  - One core active dataset: `PhysioNetMI`.
  - Main scientific question: reliable augmented-sample certification for few-shot cross-subject EEG-FM adaptation.
  - Added `docs/RESEARCH_ROADMAP.md`, `docs/FAILURE_REVIEW_PROTOCOL.md`, and `configs/experiments/mainline_scope.yaml`.
- Started the first long-term workbench trial:
  - `workbench/20260622_steegformer_sascert_core`
  - Purpose: validate `SASCert_SoftAR_LS010` on `ST-EEGFormer-small + PhysioNetMI`.
  - Trial document: `workbench/20260622_steegformer_sascert_core/TRIAL.md`
- Began promoting reusable code into `sas_core`:
  - `sas_core/data/physionet_mi.py`
  - `sas_core/data/transforms.py`
  - `sas_core/backbones/steegformer.py`
  - `sas_core/backbones/import_utils.py`
  - `sas_core/metrics/classification.py`
  - `sas_core/utils/io.py`
  - `sas_core/utils/seed.py`
- Smoke-validated the promoted modules:
  - PhysioNetMI cache shape: `[4917, 64, 640]`
  - Splits: train `3150`, val `867`, test `900`
  - ST input conversion shape: `[2, 64, 512]`
  - ST wrapper forward shape: `[2, 2]`
  - ST checkpoint loaded keys: `101`
- Implemented the first ST-SAS-Cert workbench runner:
  - `workbench/20260622_steegformer_sascert_core/runner.py`
  - Supports smoke and mini/full target/seed runs.
  - Produces metrics, paired comparisons, score rows, compact JSON, and a markdown report.
- Ran ST-SAS-Cert smoke:
  - target `90`, seed `20`; completed end-to-end.
- Ran ST-SAS-Cert mini matrix:
  - targets `90,91,92`; seeds `20,21`
  - `SASCert_SoftAR_LS010` vs `NaiveAug_LS010`:
    - delta BAcc `+0.0255`
    - delta Macro-F1 `+0.0247`
    - delta ECE `-0.0115`
    - delta NLL `-0.0187`
    - delta Brier `-0.0124`
    - subject win rate Macro-F1 `0.3333`
    - seed win rate Macro-F1 `1.0000`
  - Interpretation: positive average signal, but subject win rate is too low for a conclusion; run the full target-subject x seed matrix before making a decision.
- Ran ST-SAS-Cert full original-pretrained-feature matrix:
  - targets `90-109`; seeds `20,21,22,23,24`
  - rows `400`
  - `SASCert_SoftAR_LS010` vs `NaiveAug_LS010`:
    - delta BAcc `+0.0121`
    - delta Macro-F1 `+0.0116`
    - delta ECE `-0.0030`
    - delta NLL `-0.0011`
    - delta Brier `-0.0002`
    - subject win rate Macro-F1 `0.25`
    - seed win rate Macro-F1 `0.40`
  - Interpretation: mean effect is positive, but full Go criteria failed. All groups are close to chance, so the first broken link is likely weak original pretrained frozen ST features for this few-shot MI protocol.
- Wrote focused failure review:
  - `workbench/20260622_steegformer_sascert_core/FAILURE_REVIEW.md`
  - `workbench/20260622_steegformer_sascert_core/failure_review_summary.json`
  - Decision: `revise_training`.
  - Next action: rerun the same SAS-Cert groups using source/validation-trained ST checkpoint `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt` as a frozen feature extractor.
- Updated `workbench/20260622_steegformer_sascert_core/runner.py`:
  - Added `--steegformer-state-dict`.
  - Added `--feature-tag`.
  - Feature caches are now tag-separated to prevent source-tuned reruns from reusing original pretrained features.
- Ran source-tuned ST-SAS-Cert smoke, mini, and full matrix:
  - Source-tuned checkpoint: `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`.
  - Output tag: `st_source_tuned_full`.
  - Load audit clean: `106/106` keys loaded, no missing/unexpected keys.
  - Full source-tuned result:
    - NaiveAug BAcc `0.7088`, Macro-F1 `0.7045`.
    - SoftWeight no reject BAcc `0.7153`, Macro-F1 `0.7109`.
    - SASCert SoftAR BAcc `0.7149`, Macro-F1 `0.7108`.
    - SASCert vs Naive: BAcc `+0.0061`, Macro-F1 `+0.0063`, ECE `-0.0023`, NLL `-0.0044`.
    - Subject win rate Macro-F1 `0.25`; seed win rate Macro-F1 `0.00`.
  - Interpretation:
    - Source-tuned ST features fix the near-chance substrate problem.
    - Locked SoftAR still does not satisfy reliability criteria.
    - Immediate next branch is not a new model/dataset; it is artifact-gate diagnosis and a cleaner `SoftWeight_noReject` branch under the same ST setup.
- Completed artifact-gate diagnosis and p80 mini check:
  - Diagnostics show p90 artifact gate is precise but conservative.
  - p80 catches all synthetic BadArtifact without rejecting clean candidates in score-space, but mini training hurts Macro-F1 and BAcc.
  - p70 starts rejecting clean samples and is rejected.
  - Current ST branch decision: `prefer_softweight_no_reject_for_st_classification; keep_artifact_gate_as_calibration_diagnostic`.
- Added ST method decision and next diagnostic configs:
  - `workbench/20260622_steegformer_sascert_core/METHOD_DECISION.md`
  - `workbench/20260622_steegformer_sascert_core/summarize_group_comparison.py`
  - `configs/experiments/steegformer_physionetmi_softweight_no_reject_confirm.yaml`
  - `configs/experiments/steegformer_physionetmi_subject_heterogeneity.yaml`
  - Mainline config now points to subject heterogeneity diagnostic as the active next ST step.
- Ran subject heterogeneity diagnostic from existing source-tuned full outputs:
  - Output report: `workbench/20260622_steegformer_sascert_core/outputs/subject_heterogeneity/SUBJECT_HETEROGENEITY_REPORT.md`
  - Main finding: SoftWeight has positive mean Macro-F1 but only `0.60` subject win rate; gains are stronger for lower-baseline subjects (`r=-0.4209` with baseline Macro-F1).
  - Mainline config now points to `configs/experiments/steegformer_physionetmi_support_routing_dev.yaml`.
- Ran validation-subject support-routing development:
  - Output report: `workbench/20260622_steegformer_sascert_core/outputs/support_routing_dev/SUPPORT_ROUTING_DEV_REPORT.md`
  - LOSO support routing failed to beat the best constant dev strategy.
  - Mainline config now points to `configs/experiments/cbramod_physionetmi_sascert_matched.yaml`.
- Created the CBraMod PhysioNetMI matched-validation workbench:
  - `workbench/20260622_cbramod_physionetmi_sascert_matched`
  - Added `TRIAL.md`, `status.json`, and `config.yaml`.
  - This is the next active mainline branch.
- Implemented and ran CBraMod PhysioNetMI smoke:
  - Added reusable wrapper `sas_core/backbones/cbramod.py`.
  - Added smoke script `workbench/20260622_cbramod_physionetmi_sascert_matched/01_smoke_cbramod_physionetmi.py`.
  - Smoke passed with pooled feature shape `[8,200]`, no NaN/Inf, and full checkpoint load `211/211`.
- Built full frozen CBraMod feature cache:
  - Added `workbench/20260622_cbramod_physionetmi_sascert_matched/02_cache_cbramod_features.py`.
  - Cache shape `[4917,200]`, no NaN/Inf, no raw trial arrays saved.
  - Next step is CBraMod mini matrix.
- Ran CBraMod PhysioNetMI matched mini matrices and repair diagnostics:
  - Current SAS-Cert mini did not justify full expansion:
    - NaiveAug BAcc `0.5349`, Macro-F1 `0.4983`, ECE `0.0843`.
    - SASCert SoftAR BAcc `0.5473`, Macro-F1 `0.4988`, ECE `0.1120`.
    - SAS vs Naive: BAcc `+1.24pp`, Macro-F1 `+0.05pp`, ECE `+2.77pp`.
  - Failure review found current mixed-bad scalar SAS score is directionally wrong on PhysioNetMI:
    - current SAS clean-vs-bad AUC `0.1969`.
    - direction-fixed total AUC `0.8911`.
    - artifact-gate physio AUC `0.9022`.
  - Repaired CBraMod mini improved classification but failed calibration:
    - RepairedSoftWeight vs Naive: BAcc `+1.11pp`, Macro-F1 `+4.26pp`, ECE `+2.27pp`.
  - Support-only temperature scaling did not repair calibration:
    - RepairedTemp vs Naive: Macro-F1 `+4.26pp`, ECE `+2.21pp`, NLL slightly worse.
  - Decision: `park_cbramod_physionetmi_full_expansion`.
- Ran cross-backbone certificate direction audit:
  - Workbench: `workbench/20260622_cross_backbone_cert_direction_audit`.
  - Scope: existing score rows only; targets `90,91,92`; seeds `20,21`; no retraining and no new augmentation generation.
  - Main finding:
    - Current scalar `sas_score` is directionally wrong on both backbones:
      - CBraMod AUC `0.1969`.
      - ST AUC `0.1662`.
    - `score_artifact_gate_physio` is the strongest score-only diagnostic:
      - CBraMod AUC `0.9022`.
      - ST AUC `0.9022`.
    - `content_score` is backbone/bad-type dependent:
      - BadContent ST AUC `0.2630`.
      - BadContent CBraMod AUC `0.9025`.
  - Decision: `revise_scalar_score_before_training_expansion`.
- Defined and score-validated component-gated certificate rule v1:
  - Workbench: `workbench/20260622_component_gated_cert_rule`.
  - Rule:
    - `artifact_gate_pass = artifact_risk < fold_p90`.
    - `base = 0.75 * physio_score + 0.25 * style_score`.
    - `component_gated_v1 = ranknorm(base) * artifact_gate_pass`.
    - `content_score` is diagnostic warning only.
  - Score-only results:
    - Current scalar SAS AUC: CBraMod `0.1969`, ST `0.1662`.
    - `component_gated_v1` AUC: CBraMod `0.8395`, ST `0.8395`.
    - `score_artifact_gate_physio` AUC: CBraMod `0.9022`, ST `0.9022`.
  - Decision: `component_gated_v1_defined_score_validated`.
- Implemented component-gated ST training mode in the existing runner:
  - File changed: `workbench/20260622_steegformer_sascert_core/runner.py`.
  - Added `--experiment component_gated`.
  - Added groups:
    - `ArtifactGatePhysio_LS010`.
    - `ComponentGatedV1_LS010`.
  - Added score-row fields:
    - `artifact_gate_pass`.
    - `score_artifact_gate_physio`.
    - `component_gated_v1`.
    - `component_gated_v1_soft`.
- Ran component-gated ST reliability mini:
  - Command used source-tuned frozen ST checkpoint and mini scope targets `90,91,92`, seeds `20,21`.
  - Output report:
    - `workbench/20260622_steegformer_sascert_core/outputs/COMPONENT_GATED_ST_RELIABILITY_MINI.md`.
  - Main metrics:
    - NaiveAug BAcc `0.7869`, Macro-F1 `0.7862`, ECE `0.1413`, NLL `0.5065`.
    - SoftWeight no reject BAcc `0.7902`, Macro-F1 `0.7898`, ECE `0.1471`, NLL `0.4908`.
    - SASCert SoftAR BAcc `0.7860`, Macro-F1 `0.7852`, ECE `0.1459`, NLL `0.4966`.
    - ArtifactGatePhysio BAcc `0.7823`, Macro-F1 `0.7817`, ECE `0.1385`, NLL `0.5048`.
    - ComponentGatedV1 BAcc `0.7823`, Macro-F1 `0.7817`, ECE `0.1473`, NLL `0.5093`.
  - Key deltas:
    - ComponentGatedV1 vs Naive: Macro-F1 `-0.46pp`, ECE `+0.60pp`.
    - ComponentGatedV1 vs SoftWeight no reject: Macro-F1 `-0.81pp`, NLL `+0.0185`.
  - Decision:
    - `do_not_expand_component_gated_or_artifact_gate_physio_on_st`.
    - Score-only certificate quality does not automatically transfer to training utility.
    - The current best ST training candidate remains `SoftWeight_noReject_LS010`.
  - Mainline config now points to `configs/experiments/steegformer_physionetmi_softweight_no_reject_confirm.yaml`.
- Completed ST SoftWeight no-reject locked confirmation:
  - Added reusable summarizer:
    - `workbench/20260622_steegformer_sascert_core/summarize_softweight_locked_confirm.py`.
  - Generated locked confirm outputs:
    - `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/ST_SOFTWEIGHT_NO_REJECT_LOCKED_CONFIRM.md`.
    - `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/compact_softweight_locked_confirm.json`.
    - `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/softweight_vs_naive_subject_table.csv`.
  - Full target result for `SoftWeight_noReject_LS010` vs `NaiveAug_LS010`:
    - BAcc delta `+0.0065`.
    - Macro-F1 delta `+0.0064`.
    - ECE delta `+0.0003`.
    - NLL delta `-0.0022`.
    - Brier delta `-0.0057`.
    - Positive subject mean-delta rate `0.60`.
    - Majority-seed subject win rate `0.15`.
    - Seed win rate `0.00`.
  - Decision:
    - `do_not_promote_softweight_no_reject`.
    - Positive mean signal exists, but reliability gates fail.
- Completed focused SoftWeight failure review:
  - Added:
    - `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/SOFTWEIGHT_FAILURE_REVIEW.md`.
    - `workbench/20260622_steegformer_sascert_core/outputs/locked_confirm/softweight_failure_review_summary.json`.
    - `configs/experiments/st_score_weight_training_mismatch_failure_synthesis.yaml`.
  - Failure review decision:
    - `revise_training`.
  - First broken link:
    - clean-vs-bad certificate quality does not reliably translate into subject/seed-stable training utility.
  - Next allowed diagnostic:
    - `support_candidate_utility_alignment_audit`.
  - Stop rule:
    - run only one existing-output utility-alignment audit before either designing a locked support-only weighting/routing rule or parking ST weighting variants.
  - Mainline config now points to `configs/experiments/st_score_weight_training_mismatch_failure_synthesis.yaml`.
- Ran the single allowed support/candidate utility-alignment audit:
  - Added script:
    - `workbench/20260622_steegformer_sascert_core/audit_support_candidate_utility_alignment.py`.
  - Outputs:
    - `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/SUPPORT_CANDIDATE_UTILITY_ALIGNMENT_AUDIT.md`.
    - `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/compact_utility_alignment_audit.json`.
    - `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/fold_utility_alignment_features.csv`.
    - `workbench/20260622_steegformer_sascert_core/outputs/utility_alignment_audit/utility_alignment_correlations.csv`.
  - Important correction during execution:
    - Initial correlation included retrospective outcome variables, which are not legal future predictors.
    - Script was corrected so the decision uses candidate-only score summaries.
  - Final result:
    - strongest candidate-only Spearman with SoftWeight benefit: `0.1168`.
    - strongest feature: `clean_artifact_risk_raw_mean`.
    - actionable threshold: `0.35`.
    - strong alignment found: `false`.
  - Decision:
    - `park_st_weighting_variants`.
    - Do not continue gate/weight search on ST without a new method hypothesis.
- Reframed the main SAS-Cert route:
  - Added:
    - `docs/SAS_CERT_DIAGNOSTIC_REFRAME.md`.
    - `configs/experiments/sas_cert_diagnostic_reframe_after_weighting_failures.yaml`.
    - `configs/experiments/next_mve_after_diagnostic_reframe.yaml`.
  - New framing:
    - SAS-Cert is currently best supported as a diagnostic reliability certificate for augmentation candidates.
    - It detects harmful augmentation modes and score-direction failures.
    - Current weighting/rejection policies are not reliable enough to be promoted as deployable training methods.
  - Next planned MVE:
    - `sas_cert_diagnostic_certificate_pack_physionetmi`.
    - Existing outputs only; no new training, backbone, or dataset.
  - Mainline config now points to `configs/experiments/sas_cert_diagnostic_reframe_after_weighting_failures.yaml`.
- Completed the PhysioNetMI diagnostic certificate pack MVE:
  - Added script:
    - `scripts/summarize_diagnostic_certificate_pack.py`.
  - Fixed config input paths to use existing CBraMod compact outputs.
  - Generated:
    - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PACK_PHYSIONETMI.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/compact_result.json`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/diagnostic_auc_summary.csv`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/training_policy_summary.csv`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/gate_summary.csv`.
  - Diagnostic AUC evidence:
    - CBraMod current scalar SAS AUC `0.1969`; component-gated v1 AUC `0.8395`; artifact-gate physio AUC `0.9022`.
    - ST current scalar SAS AUC `0.1662`; component-gated v1 AUC `0.8395`; artifact-gate physio AUC `0.9022`.
  - Training policy evidence:
    - CBraMod repaired score: Macro-F1 `+4.26pp`, but ECE `+2.27pp`, not promoted.
    - ST SoftWeight no-reject: Macro-F1 `+0.64pp`, but subject/seed reliability failed, not promoted.
  - Gates:
    - diagnostic AUC gate `true`.
    - scalar failure gate `true`.
    - protocol gate `true`.
    - weighting-policy non-promotion gate `true`.
  - Decision:
    - `write_diagnostic_certificate_paper_path`.
  - Next action:
    - `prepare_diagnostic_certificate_paper_outline`.
- Created the diagnostic certificate paper outline:
  - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PAPER_OUTLINE.md`.
  - Core claim:
    - SAS-Cert is currently supported as a diagnostic reliability certificate, not as a deployable weighting/rejection training policy.
  - Minimal remaining evidence checklist:
    - bad-type component AUC table.
    - protocol-leakage audit table.
    - causal chain diagram.
    - augmentation failure-mode definitions.
    - file-path traceability for every reported number.
  - Mainline config now points to `prepare_diagnostic_certificate_paper_outline`.
- Completed the diagnostic paper evidence checklist:
  - Extended `scripts/summarize_diagnostic_certificate_pack.py`.
  - Generated:
    - `docs/SAS_CERT_DIAGNOSTIC_EVIDENCE_CHECKLIST.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/bad_type_component_auc.csv`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/protocol_leakage_audit.csv`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/failure_mode_definitions.csv`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/number_traceability.csv`.
  - Updated `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_PAPER_OUTLINE.md` checklist to completed.
- Created the paper draft plan:
  - `configs/experiments/diagnostic_certificate_paper_draft_plan.yaml`.
  - Next active task:
    - create `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT.md`.
  - Mainline config now points to `diagnostic_certificate_paper_draft_plan`.
- Created the first conservative diagnostic certificate paper draft:
  - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT.md`.
  - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/claim_support_table.csv`.
  - Draft stance:
    - Supported: SAS-Cert is a diagnostic reliability certificate.
    - Unsupported: SAS-Cert is a reliable deployable augmentation weighting/rejection method.
  - The draft includes introduction, problem definition, diagnostic framework,
    protocol, results, discussion, limitations, conclusion, and evidence index.
- Created the draft review plan:
  - `configs/experiments/diagnostic_certificate_draft_review_plan.yaml`.
  - Mainline config now points to `diagnostic_certificate_draft_review_plan`.
- Completed the diagnostic certificate draft review:
  - Added:
    - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_REVIEW.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/tables/draft_revision_checklist.csv`.
  - Review decision:
    - `draft_review_passed_with_minor_scope_revision`.
  - Revision applied:
    - tightened all diagnostic AUC wording to the current synthetic mixed
      bad-augmentation diagnostic pool.
  - Confirmed:
    - no promotion of weighting/rejection policies.
    - key numeric claims trace to `claim_support_table.csv` and evidence tables.
- Created the manuscript polish and citation plan:
  - `configs/experiments/manuscript_polish_and_citation_plan.yaml`.
  - Next active task:
    - create `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md`.
    - create `docs/SAS_CERT_CITATION_PLAN.md`.
    - create `docs/SAS_CERT_FIGURE_TABLE_PLAN.md`.
  - Mainline config now points to `manuscript_polish_and_citation_plan`.
- Completed the manuscript polish and citation package:
  - Created:
    - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_DRAFT_POLISHED.md`.
    - `docs/SAS_CERT_CITATION_PLAN.md`.
    - `docs/SAS_CERT_FIGURE_TABLE_PLAN.md`.
  - Marked `configs/experiments/manuscript_polish_and_citation_plan.yaml`
    as completed.
  - Added next review config:
    - `configs/experiments/diagnostic_manuscript_review_next.yaml`.
  - Updated `configs/experiments/mainline_scope.yaml` and
    `docs/RESEARCH_ROADMAP.md` so the active next step is a manuscript package
    review, not a new experiment.
  - No experiments were run and no new training claims were added.
- Completed the diagnostic manuscript package review:
  - Review report:
    - `docs/SAS_CERT_DIAGNOSTIC_MANUSCRIPT_PACKAGE_REVIEW.md`.
  - Marked:
    - `configs/experiments/diagnostic_manuscript_review_next.yaml` as completed.
  - Added next active plan:
    - `configs/experiments/bibliography_and_figure_generation_plan.yaml`.
  - Mainline config now points to bibliography and figure/table generation from
    existing evidence only.
  - Verdict:
    - `package_review_passed_with_submission_preparation_gaps`.
  - Passed checks:
    - citation placeholders mapped to local anchors.
    - figure/table plans have existing source files.
    - claim boundaries match `claim_support_table.csv`.
    - current weighting/rejection policies remain explicitly not promoted.
  - Remaining writing/package gaps:
    - resolve BibTeX placeholders.
    - remove the project-management citation-placeholder sentence from the
      abstract before submission.
    - render manuscript tables/figures from existing CSV/JSON only.
- Completed bibliography trace and manuscript asset generation:
  - Added generator:
    - `scripts/generate_diagnostic_manuscript_assets.py`.
  - Generated:
    - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT.md`.
    - `docs/SAS_CERT_BIBLIOGRAPHY_TRACE.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_asset_manifest.json`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table1_protocol_and_claim_boundary.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table2_diagnostic_auc_summary.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table3_bad_type_component_auc.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table4_training_policy_non_promotion.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/table5_claim_support.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_tables/appendix_protocol_leakage_audit.md`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure1_certificate_overview.svg`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure2_diagnostic_auc.svg`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure3_component_specificity_heatmap.svg`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure4_training_policy_non_promotion.svg`.
    - `outputs/runs/sas_cert_diagnostic_certificate_pack_physionetmi/manuscript_figures/figure5_causal_chain.svg`.
  - Marked:
    - `configs/experiments/bibliography_and_figure_generation_plan.yaml` as completed.
  - Added next active review:
    - `configs/experiments/submission_package_quality_review.yaml`.
  - Mainline config now points to submission package quality review.
  - No experiments, new training, or new scientific claims were introduced.
- Completed submission package quality review:
  - Review report:
    - `docs/SAS_CERT_SUBMISSION_PACKAGE_QUALITY_REVIEW.md`.
  - Verified:
    - manifest outputs exist.
    - 6 table drafts and 5 SVG figure drafts are present.
    - all 5 SVG files parse.
    - submission draft no longer contains internal output/workbench paths,
      `Citation placeholders`, or `Evidence Index`.
    - current weighting/rejection policy remains explicitly not promoted.
  - Fixed during review:
    - `scripts/generate_diagnostic_manuscript_assets.py` now removes internal
      evidence-path lines from the submission draft.
    - Table 1 now names the prohibited claims instead of only saying
      `do not claim`.
  - Marked:
    - `configs/experiments/submission_package_quality_review.yaml` as completed.
  - Added next active plan:
    - `configs/experiments/bibtex_latex_figure_polish_plan.yaml`.
  - Mainline config now points to bibliography, LaTeX, and figure/table polish.
  - No new experiments, training, data, or scientific claims were introduced.
- Completed BibTeX, LaTeX, and figure/table polish pass:
  - Updated generator:
    - `scripts/generate_diagnostic_manuscript_assets.py`.
  - Generated:
    - `docs/SAS_CERT_DIAGNOSTIC_CERTIFICATE_SUBMISSION_DRAFT_LATEX.md`.
    - `docs/SAS_CERT_REFERENCES.bib`.
    - `docs/SAS_CERT_BIBLIOGRAPHY_RESOLUTION_REPORT.md`.
    - `docs/SAS_CERT_FIGURE_TABLE_POLISH_REPORT.md`.
  - Manifest now includes:
    - `latex_submission_draft`.
    - `references_bib`.
    - `bibliography_resolution_report`.
    - `figure_table_polish_report`.
  - Verification:
    - 13 BibTeX entries.
    - BibTeX braces balanced.
    - ST-EEGFormer-small citation resolved from local README BibTeX.
    - no unresolved `[RE...]` / bracket-style manuscript citations in the
      LaTeX draft.
    - 5 SVG figures parse.
    - all manifest outputs exist.
  - Marked:
    - `configs/experiments/bibtex_latex_figure_polish_plan.yaml` as completed.
  - Added next active review:
    - `configs/experiments/submission_readiness_review.yaml`.
  - Mainline config now points to final submission-readiness review.
  - No new experiments, training, data, or scientific claims were introduced.
- Completed final submission-readiness review:
  - Review report:
    - `docs/SAS_CERT_SUBMISSION_READINESS_REVIEW.md`.
  - Verified:
    - all `\cite{}` keys in the LaTeX draft resolve to
      `docs/SAS_CERT_REFERENCES.bib`.
    - BibTeX has 13 entries and balanced braces.
    - all manifest outputs exist.
    - 5 SVG figures parse.
    - 6 Markdown table drafts exist.
    - LaTeX draft preserves the diagnostic-certificate claim boundary and does
      not promote weighting/rejection policies.
  - Noted non-blocking cleanup:
    - three unused BibTeX entries remain and should be removed or cited in the
      venue package.
  - Marked:
    - `configs/experiments/submission_readiness_review.yaml` as completed.
  - Added next active plan:
    - `configs/experiments/venue_specific_formatting_plan.yaml`.
  - Mainline config now points to a venue-agnostic LaTeX paper package.
  - No new experiments, training, data, or scientific claims were introduced.
- Completed venue-agnostic LaTeX paper package generation:
  - Added reproducible package builder:
    - `scripts/build_sas_cert_paper_package.py`.
  - Generated paper package:
    - `paper/sas_cert_diagnostic_certificate/main.tex`.
    - `paper/sas_cert_diagnostic_certificate/references.bib`.
    - `paper/sas_cert_diagnostic_certificate/README.md`.
    - `paper/sas_cert_diagnostic_certificate/figures/`.
    - `paper/sas_cert_diagnostic_certificate/tables/`.
  - Generated formatting report:
    - `docs/SAS_CERT_VENUE_FORMATTING_REPORT.md`.
  - Verification:
    - 10 cited BibTeX keys resolve inside the package bibliography.
    - package bibliography has no unused entries.
    - 5 SVG figures and 4 generated LaTeX inline tables are present.
    - 6 Markdown evidence tables were copied for traceability.
    - `main.tex` does not expose internal `workbench/` or `outputs/runs/`
      paths.
    - local compile was not run because `latexmk` and `pdflatex` are not
      available in this environment.
  - Marked:
    - `configs/experiments/venue_specific_formatting_plan.yaml` as completed.
  - Added next active review:
    - `configs/experiments/venue_package_integrity_review.yaml`.
  - Mainline config now points to venue package integrity review.
  - No new experiments, training, data, or scientific claims were introduced.
- Completed venue package integrity review:
  - Integrity report:
    - `docs/SAS_CERT_VENUE_PACKAGE_INTEGRITY_REVIEW.md`.
  - Generator fixes applied:
    - `scripts/build_sas_cert_paper_package.py` now strips source-authored
      section numbers before LaTeX auto-numbering.
    - generated inline tables now use text-safe alignment and
      `\resizebox{\linewidth}{!}{...}`.
    - `graphicx` is included in generated `main.tex`.
  - Verification:
    - 10 cited keys, 0 missing BibTeX entries.
    - 0 unused entries in the package bibliography.
    - 5 SVG figure paths resolve.
    - 4 LaTeX table inputs resolve.
    - document, itemize, and verbatim environment counts are balanced.
    - no Markdown residue, `[RE...]` placeholders, `workbench/`, or
      `outputs/runs/` paths remain in `main.tex`.
    - local PDF compilation and ChkTeX lint were not run because the tools are
      unavailable in the current environment.
  - Marked:
    - `configs/experiments/venue_package_integrity_review.yaml` as completed.
  - Added next active plan:
    - `configs/experiments/venue_template_selection_plan.yaml`.
  - Mainline config now points to venue/template selection and adaptation.
  - No new experiments, training, data, or scientific claims were introduced.
- Completed venue/template route selection:
  - Selected default route:
    - `arxiv_article_first`.
  - Added arXiv-first venue package builder:
    - `scripts/build_sas_cert_arxiv_package.py`.
  - Generated venue package:
    - `paper/sas_cert_diagnostic_certificate_venue/main.tex`.
    - `paper/sas_cert_diagnostic_certificate_venue/references.bib`.
    - `paper/sas_cert_diagnostic_certificate_venue/tables/`.
    - `paper/sas_cert_diagnostic_certificate_venue/figures_svg_original/`.
  - Venue report:
    - `docs/SAS_CERT_VENUE_SELECTION_REPORT.md`.
  - Verification:
    - venue `main.tex` has no `svg` package dependency.
    - 5 guarded `\includegraphics` figure slots are present.
    - cited BibTeX keys resolve and no package BibTeX entries are unused.
    - no `[RE...]`, `workbench/`, or `outputs/runs/` paths remain in venue
      `main.tex`.
  - Environment limitation:
    - SVG-to-PDF conversion was not performed because `inkscape`,
      `rsvg-convert`, and Python `cairosvg` are unavailable.
  - Marked:
    - `configs/experiments/venue_template_selection_plan.yaml` as completed.
  - Added next active plan:
    - `configs/experiments/arxiv_figure_conversion_compile_plan.yaml`.
  - Mainline config now points to arXiv figure conversion / compile readiness.
  - No new experiments, training, data, or scientific claims were introduced.

### 2026-06-23 Execution Environment Permission Check

- User asked to verify whether the runtime is sandboxed.
- Checked:
  - current directory: `/ai/224duibishiyan/615新研究`
  - process user: `root`
  - `id`: `uid=0(root) gid=0(root) groups=0(root)`
  - project path is writable; a temporary `.codex_permission_check.tmp` file was
    created and immediately removed.
- Conclusion:
  - The current execution environment is not filesystem-sandboxed from the tool
    perspective and has root-level permissions.
- Safety note:
  - Continue to avoid destructive commands, raw-data copies, and unrelated file
    rewrites unless explicitly requested.

### 2026-06-23 arXiv Figure Conversion And Compile Readiness

- Active plan completed:
  - `configs/experiments/arxiv_figure_conversion_compile_plan.yaml`.
- Added figure generator:
  - `scripts/generate_arxiv_pdf_figures.py`.
- Generated five PDF figures from existing evidence tables only:
  - `paper/sas_cert_diagnostic_certificate_venue/figures/figure1_certificate_overview.pdf`
  - `paper/sas_cert_diagnostic_certificate_venue/figures/figure2_diagnostic_auc.pdf`
  - `paper/sas_cert_diagnostic_certificate_venue/figures/figure3_component_specificity_heatmap.pdf`
  - `paper/sas_cert_diagnostic_certificate_venue/figures/figure4_training_policy_non_promotion.pdf`
  - `paper/sas_cert_diagnostic_certificate_venue/figures/figure5_causal_chain.pdf`
- Compile/readiness report:
  - `docs/SAS_CERT_ARXIV_COMPILE_REPORT.md`.
- Verification:
  - all 5 generated files have valid `%PDF-1.4` headers.
  - venue `main.tex` has 5 guarded figure slots and 0 missing referenced PDF
    figures.
  - cited BibTeX keys resolve and no package BibTeX entries are unused.
  - no `[RE...]`, `workbench/`, or `outputs/runs/` paths remain in venue
    `main.tex`.
  - Python syntax checks passed for the paper/venue build scripts.
- Limitation:
  - local PDF compilation was not run because `latexmk`, `pdflatex`,
    `xelatex`, `lualatex`, `tectonic`, and `chktex` are unavailable.
- Added next active plan:
  - `configs/experiments/arxiv_latex_tooling_or_external_compile_plan.yaml`.
- Mainline config now points to obtaining a trustworthy LaTeX compile result,
  either through local tooling or an external compile environment.
- No new experiments, training runs, data copies, thresholds, or scientific
  claims were introduced.

### 2026-06-23 arXiv Local LaTeX Compile

- Active plan completed:
  - `configs/experiments/arxiv_latex_tooling_or_external_compile_plan.yaml`.
- Installed local LaTeX compile tooling via `apt-get`:
  - `latexmk`
  - `texlive-latex-base`
  - `texlive-latex-recommended`
  - `texlive-latex-extra`
  - `texlive-fonts-recommended`
  - `chktex`
- Tooling verified:
  - `latexmk` 4.67
  - `pdflatex` pdfTeX 3.14159265-2.6-1.40.20
  - `bibtex` 0.99d
  - `chktex` 1.7.6
- Applied compile-polish fixes through the arXiv package builder:
  - central research question changed from overlong display math to an
    emphasized quote.
  - long `clean_artifact_risk_raw_mean` text token made breakable.
- Compile command:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  - run from `paper/sas_cert_diagnostic_certificate_venue`.
- Compile result:
  - generated `paper/sas_cert_diagnostic_certificate_venue/main.pdf`.
  - final PDF size: `271202` bytes.
  - final PDF pages: `11`.
  - final log counts: `LaTeX Warning=0`, `Overfull=0`, `Undefined=0`,
    `Error=0`, `Emergency stop=0`, `Fatal=0`.
  - ChkTeX style warnings: `31`, non-blocking typography/style warnings.
- Final compile report:
  - `docs/SAS_CERT_ARXIV_FINAL_COMPILE_REPORT.md`.
- Marked:
  - `configs/experiments/arxiv_latex_tooling_or_external_compile_plan.yaml` as
    completed.
- Added next active plan:
  - `configs/experiments/arxiv_submission_bundle_review_plan.yaml`.
- Mainline config now points to arXiv submission bundle and author metadata
  review.
- No new experiments, training runs, data copies, thresholds, or scientific
  claims were introduced.

### 2026-06-23 Conversation Ledger Maintenance Rule

- User clarified that the long-term project ledger must be updated after every
  answered request:
  - `/ai/224duibishiyan/615新研究/PROJECT_MANAGEMENT.md`
- Standing rule updated accordingly:
  - after each answered user request, record the decision, context learned,
    actions taken, and explicit no-op status when no research files,
    experiments, or outputs were changed.
- Reason for previous omission:
  - "understand, but do not do anything" was interpreted as no project-file
    edits; however, the long-term ledger is a persistent maintenance file and
    should still receive a no-op/context update.
- Context learned from recent user-supplied AI notes:
  - Innovation positioning should emphasize augmentation-level reliability
    certification and reliable utilization, not a new augmentation operator,
    artifact rejection method, or generic sample reweighting method.
  - SAS-Cert-EEG should be framed as a risk-controlled augmentation policy:
    safety gate -> label-preservation evidence -> utility weighting /
    calibration-aware training.
  - The fixed scalar `sas_score` should not be promoted as a universal ranker;
    the project direction remains a multi-dimensional reliability certificate
    with component-gated decision logic.
- Current no-op status:
  - No experiments, training runs, data copies, paper content changes, figures,
    thresholds, or scientific result files were modified in this context-update
    turn.

### 2026-06-23 SAS-Cert v1.1 Reference-Tool Blueprint Context

- User asked to understand a new AI note but not take implementation action yet,
  while keeping the project-management ledger updated.
- Context learned:
  - The proposed next algorithm framing is `SAS-Cert-SoftAR-LS v1.1`.
  - The method should remain centered on the scientific question, not on a
    pile-up of external modules.
  - External EEG tools should support specific decision layers:
    - `Autoreject`, `MNE-ICALabel`, and `MNE-Features` for safety/artifact
      evidence.
    - `ST-EEGFormer-small` and `CBraMod` embeddings for content and prototype
      evidence.
    - `pyRiemann` and `MNE-Features` for covariance, spectral, and statistical
      physiology/style plausibility.
    - `Braindecode` as a possible augmentation/baseline utility.
    - `EEG-DLite`, `MOABB`, Channel Reflection, HAPPE/PREP/ADJUST, and
      ArtifactGen as later/cite/audit candidates rather than immediate
      mainline dependencies.
  - The intended v1.1 logic is:
    - safety gate -> label-preservation evidence -> physiology/style
      plausibility -> utility weight -> calibration-aware training.
  - The proposed first actual experiment, if later approved, would still be
    bounded to `ST-EEGFormer-small + PhysioNetMI` with four groups:
    `NaiveAug_LS010`, `ArtifactReject_LS010`, `SoftWeight_noReject_LS010`,
    and `SAS-Cert-SoftAR-LS-v1.1`.
- Decision:
  - Treat this as design context only for now.
  - Do not download reference repositories, run smoke tests, create
    `outputs/reference_algorithm_audit`, or change SAS-Cert code until the user
    explicitly asks for that audit/implementation task.
- Current no-op status:
  - No experiments, training runs, repository downloads, data copies, paper
    content changes, figures, thresholds, or scientific result files were
    modified in this turn.

### 2026-06-23 SAS-Cert Direct-Use Reference Algorithm Audit

- Task completed:
  - `SASCERT_DIRECT_USE_ALGORITHM_AUDIT_AND_V1_BLUEPRINT`
- Purpose:
  - Download and audit whitelisted, directly reusable EEG algorithm/tool
    projects for `SAS-Cert-SoftAR-LS v1.1`.
  - Keep the method organized as:
    - safety gate -> label-preservation evidence -> physiology/style
      plausibility -> utility weight -> calibration-aware training.
- Runner:
  - `workbench/20260623_reference_algorithm_audit/01_run_reference_algorithm_audit.py`
- Command:
  - `python3 workbench/20260623_reference_algorithm_audit/01_run_reference_algorithm_audit.py`
- Isolation:
  - `python3 -m venv` failed because `ensurepip` / `python3.8-venv` is not
    available.
  - The runner was revised to use isolated `pip --target` installation under:
    - `outputs/reference_algorithm_audit/python_target`
  - No packages were installed into the main Python environment.
- Third-party reference code directory:
  - `third_party/reference_algorithms`
- Downloaded / inspected projects:
  - `MNE-ICALabel`: `third_party/reference_algorithms/mne-icalabel`
  - `Autoreject`: `third_party/reference_algorithms/autoreject`
  - `pyRiemann`: `third_party/reference_algorithms/pyRiemann`
  - `MNE-Features`: `third_party/reference_algorithms/mne-features`
  - `Braindecode`: `third_party/reference_algorithms/braindecode`
  - `EEG-DLite`: `third_party/reference_algorithms/EEG-DLite`
  - `MOABB`: `third_party/reference_algorithms/moabb`
  - Channel Reflection official-candidate sparse clone:
    `third_party/reference_algorithms/EEGAug_ChannelReflection_sparse`
- Channel Reflection source note:
  - Web search found `https://github.com/wzwvv/EEGAug`, whose README states it
    is the official implementation of Channel Reflection.
  - The arXiv HTML v1 page also footnotes
    `https://github.com/sylyoung/DeepTransferEEG`.
  - The audit used `wzwvv/EEGAug` and sparse-cloned only README/code/config
    files, excluding `data/` to avoid copying EEG data.
- Audit result:
  - `use_now`:
    - `pyRiemann`: import and synthetic covariance/Riemannian-distance smoke
      passed.
    - `MNE-Features`: import and synthetic feature extraction smoke passed
      after isolated `PyWavelets` install.
  - `use_offline_only`:
    - `Autoreject`: import and tiny MNE Epochs reject-log smoke passed after
      isolated `h5io` install.
  - `use_later`:
    - `EEG-DLite`: cloned and inspected; has `distillate_datasets.py`, but is
      better suited to later outlier/redundancy filtering.
    - `MOABB`: import and PhysionetMI dataset-name discovery smoke passed, but
      remains benchmark/protocol reference only.
    - Channel Reflection: sparse clone succeeded without `data/`; useful later
      as a knowledge-driven augmentation-pool candidate.
  - `blocked`:
    - `MNE-ICALabel`: package body installed in isolated target, but import
      failed due current MNE API/version mismatch (`mne.io.Info` import).
    - `Braindecode`: package body plus `skorch` installed in isolated target,
      but direct import failed on heavier transitive dependency stack; do not
      force into v1.1.
- Primary outputs:
  - `outputs/reference_algorithm_audit/REFERENCE_ALGORITHM_AUDIT_REPORT.md`
  - `outputs/reference_algorithm_audit/reference_algorithm_inventory.csv`
  - `outputs/reference_algorithm_audit/reference_algorithm_inventory.json`
  - `outputs/reference_algorithm_audit/sascert_softar_ls_v1_1_blueprint.md`
  - `outputs/reference_algorithm_audit/compact_reference_algorithm_result.json`
  - `outputs/reference_algorithm_audit/audit_command_logs.json`
- Recommended v1.1 implementation structure:
  - artifact risk:
    - existing rule score plus optional Autoreject offline audit;
      MNE-ICALabel deferred until version-matched environment exists.
  - content:
    - ST/CBra embedding similarity, class prototype consistency, and prediction
      consistency.
  - physiology:
    - `pyRiemann` covariance/Riemannian distance plus MNE-Features bandpower /
      statistical features.
  - style:
    - target support mean/std, bandpower, and covariance summary; auxiliary
      weight only.
  - utility:
    - `w = 0.2 + 0.8 * ranknorm(score)` for safety-passing augmented samples;
      `w = 0` for rejected samples.
  - calibration:
    - label smoothing fixed at `0.10`.
- Next possible experiment after explicit user approval:
  - `ST-EEGFormer-small + PhysioNetMI`
  - Groups:
    - `NaiveAug_LS010`
    - `ArtifactReject_LS010`
    - `SoftWeight_noReject_LS010`
    - `SAS-Cert-SoftAR-LS-v1.1`
- Safety/compliance:
  - No model training.
  - No SAS-Cert experiment.
  - No raw EEG data copied.
  - No MIRepNet / EEGPT integration.
  - No old experiment outputs modified.

### 2026-06-23 Reference Audit Follow-Up: Transfer/Relay Clarification

- User asked whether the previously discussed "中转系统" was used and what was
  downloaded in the reference-algorithm audit.
- Clarification:
  - A project search did not find an explicit local `中转`, `relay`, `proxy`, or
    `mirror` system/script/config in the current tracked management/code
    context.
  - The reference-algorithm audit did not use a special relay/transfer system.
  - Downloads were direct, whitelist-only GitHub clones plus isolated
    `pip --target` package installs under:
    - `outputs/reference_algorithm_audit/python_target`
  - No packages were installed into the main Python environment.
- Downloaded reference repositories and intended roles:
  - `MNE-ICALabel`: artifact component probability reference; currently blocked
    by MNE API/version mismatch.
  - `Autoreject`: MNE Epochs bad-trial/bad-sensor reject-log tool; usable as
    offline safety-gate audit.
  - `pyRiemann`: covariance and Riemannian distance tool; usable now for
    physiology/style scores.
  - `MNE-Features`: spectral/statistical feature extractor; usable now for
    artifact/physiology/style features.
  - `Braindecode`: EEG augmentation/baseline utility candidate; currently
    blocked by heavier dependency stack.
  - `EEG-DLite`: outlier/redundancy filtering/data-distillation reference; use
    later.
  - `MOABB`: benchmark/data-loader protocol reference; use later.
  - Channel Reflection sparse clone: knowledge-driven augmentation candidate;
    use later, with `data/` excluded.
- Current no-op status:
  - This follow-up only updated the project ledger; no new downloads,
    experiments, training runs, data copies, code changes, paper edits, figures,
    thresholds, or scientific result files were introduced.

### 2026-06-23 Workbench Trial-Box System Clarification

- User clarified that the "中转系统" refers to the project workflow system, not
  a network relay/proxy:
  - `docs/PROJECT_WORKFLOW.md`
  - `workbench/README.md`
  - `scripts/manage_trial.py`
  - `PROJECT_INDEX.md`
  - `scripts/SCRIPT_REGISTRY.md`
  - `PROJECT_MANAGEMENT.md`
- Correct understanding:
  - The transfer/intermediate system is the `workbench/` trial-box workflow.
  - New ideas and temporary exploration should start in
    `workbench/<YYYYMMDD>_<trial_name>/`.
  - Each trial should contain `TRIAL.md`, `config.yaml`, `status.json`, and
    `outputs/`.
  - Reusable successful logic should later be promoted into `sas_core/`,
    `configs/experiments/`, or stable `scripts/`.
  - Failed trials should be summarized and archived or parked without polluting
    the mainline.
  - Long-running decisions and actions must continue to be recorded in this
    project ledger.
- Relation to the reference-algorithm audit:
  - The audit did use this workflow pattern:
    - temporary runner:
      `workbench/20260623_reference_algorithm_audit/01_run_reference_algorithm_audit.py`
    - audit reports:
      `outputs/reference_algorithm_audit/`
    - third-party code:
      `third_party/reference_algorithms/`
  - The earlier assistant answer incorrectly interpreted "中转系统" only as a
    possible network relay/proxy/mirror and missed the existing workbench
    workflow meaning.
- Future rule:
  - When the user refers to the project "中转系统", treat it as the `workbench/`
    trial-box workflow unless the context explicitly says network relay/proxy.
- Current no-op status:
  - This turn only corrected understanding and updated the project ledger; no
    new experiments, training runs, downloads, code edits outside this ledger,
    data copies, paper edits, figures, thresholds, or result files were
    introduced.

### 2026-06-23 Reference Audit Download Explanation Follow-Up

- User asked for a plain explanation of what was downloaded in the previous
  formal reference-algorithm audit and what each item is for.
- Clarification to preserve:
  - The previous formal work was the direct-use EEG reference algorithm audit
    for `SAS-Cert-SoftAR-LS v1.1`.
  - Downloaded repositories are under:
    - `third_party/reference_algorithms/`
  - Isolated Python packages are under:
    - `outputs/reference_algorithm_audit/python_target`
  - Generated audit reports are under:
    - `outputs/reference_algorithm_audit/`
  - The audit runner is under:
    - `workbench/20260623_reference_algorithm_audit/`
- Main downloaded/reference items and roles:
  - `pyRiemann`: direct-use covariance / Riemannian-distance utility for
    physiology and style plausibility scores.
  - `MNE-Features`: direct-use spectral/statistical feature extraction utility
    for artifact, physiology, and style evidence.
  - `Autoreject`: offline bad-epoch / bad-sensor safety-gate audit utility.
  - `MNE-ICALabel`: intended ICA artifact-probability reference, but currently
    blocked by local MNE API/version mismatch.
  - `Braindecode`: intended EEG augmentation/baseline utility, but currently
    blocked by heavier dependency stack.
  - `EEG-DLite`: later outlier/redundancy filtering and data-distillation
    reference.
  - `MOABB`: later benchmark/data-loader/protocol reference.
  - Channel Reflection sparse clone: later knowledge-driven augmentation
    candidate, cloned without `data/`.
- Current no-op status:
  - This follow-up only updates the project ledger and explanation context; no
    new downloads, experiments, training runs, code changes outside this
    ledger, data copies, paper edits, figures, thresholds, or result files were
    introduced.

### 2026-06-23 SAS-Cert-SoftAR-LS v1.1 ST-EEGFormer PhysioNetMI Validation

- Task completed:
  - `IMPLEMENT_AND_VALIDATE_SASCERT_SOFTAR_LS_V1_1_ON_STEEGFORMER_PHYSIONETMI`
- Workbench trial:
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi`
- Runner:
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py`
- Scope:
  - Backbone: `ST-EEGFormer-small`
  - Source-tuned checkpoint:
    `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
  - Dataset: `PhysioNetMI / EEGMMI`
  - Task: left-vs-right motor imagery
  - Runs: `R04/R08/R12`
  - Targets: `90-109`
  - Seeds: `20,21,22,23,24`
  - Support: 5-shot per class
- Candidate augmentations:
  - Gaussian noise
  - time shift
  - time crop
  - frequency mask
  - channel dropout
  - mild frequency mixup
- Implemented v1.1 decision chain:
  - Safety Gate:
    - artifact score from high-frequency energy, low-frequency drift,
      channel-energy outlier, kurtosis, skewness, and line-noise band power.
    - top 10% artifact-risk candidates rejected for gate groups.
  - Label-Preservation Evidence:
    - embedding consistency
    - class-prototype margin from source train + target support
    - prediction consistency from source-trained head KL agreement
  - Physiology / Style Plausibility:
    - `pyRiemann` covariance/Riemannian distance
    - `MNE-Features` statistics
    - bandpower deviation
    - target-support style anchor
  - Utility Weight:
    - `w = 0.2 + 0.8 * ranknorm(E_total)` for safety-passing candidates.
    - rejected candidates get `w = 0`.
  - Calibration-aware training:
    - label smoothing `0.10`.
- Tool status:
  - `pyRiemann`: used.
  - `MNE-Features`: used; numba unavailable warning is non-blocking.
  - `Autoreject`: available for offline audit, not integrated into training.
  - `MNE-ICALabel`, `Braindecode`, `EEG-DLite`, `MOABB`, and Channel Reflection
    were not introduced into this run.
- Smoke command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py --smoke --targets 90 --seeds 20 --device cuda --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt --feature-tag st_source_tuned_seed3407 --output-tag v1_1_smoke`
- Full command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 --seeds 20 21 22 23 24 --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64 --feature-batch-size 64 --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt --feature-tag st_source_tuned_seed3407 --output-tag v1_1_source_tuned_full`
- Full result:
  - `NaiveAug_LS010`:
    - BAcc `0.7156`, Macro-F1 `0.7098`, AUROC `0.7825`, ECE `0.2093`,
      NLL `0.7208`, Brier `0.4225`.
  - `ArtifactReject_LS010`:
    - BAcc `0.7074`, Macro-F1 `0.7013`, AUROC `0.7823`, ECE `0.2150`,
      NLL `0.7304`, Brier `0.4267`.
  - `SoftWeight_noReject_LS010`:
    - BAcc `0.7117`, Macro-F1 `0.7050`, AUROC `0.7823`, ECE `0.2123`,
      NLL `0.7247`, Brier `0.4238`.
  - `SAS-Cert-SoftAR-LS-v1.1`:
    - BAcc `0.7102`, Macro-F1 `0.7047`, AUROC `0.7819`, ECE `0.2105`,
      NLL `0.7275`, Brier `0.4246`.
- Primary comparison:
  - v1.1 vs Naive:
    - Delta BAcc `-0.0054`
    - Delta Macro-F1 `-0.0051`
    - Delta ECE `+0.0012`
    - Delta NLL `+0.0068`
    - Delta Brier `+0.0021`
    - Subject win rate Macro-F1 `0.25`
    - Seed win rate Macro-F1 `0.00`
  - v1.1 vs ArtifactReject:
    - Delta BAcc `+0.0028`
    - Delta Macro-F1 `+0.0033`
    - Delta ECE `-0.0045`
    - Delta NLL `-0.0028`
    - Delta Brier `-0.0022`
  - v1.1 vs SoftWeight no-reject:
    - Delta BAcc `-0.0015`
    - Delta Macro-F1 `-0.0003`
    - Delta ECE `-0.0018`
    - Delta NLL `+0.0029`
    - Delta Brier `+0.0008`
- Certificate / gate summary:
  - Mean rejected ratio for v1.1: `0.10`
  - Mean v1.1 candidate weight: `0.5351`
  - Mean score NaN/Inf count: `0.0`
  - Mean covariance NaN/Inf count: `0.0`
  - Rejection by augmentation type:
    - Gaussian noise `0.228`
    - time crop `0.156`
    - time shift `0.072`
    - frequency mask `0.048`
    - mild frequency mixup `0.049`
    - channel dropout `0.047`
- Leakage audit:
  - `passed`
  - target test was not used for artifact thresholds, ranknorm, prototypes,
    style anchor, best epoch, or best seed.
  - target test was used for final evaluation only.
- Decision:
  - `do_not_promote_v1_1_on_st`
  - v1.1 does not beat `NaiveAug_LS010`.
  - v1.1 beats standalone `ArtifactReject_LS010`, indicating the soft utility
    weighting partly repairs gate-only training.
  - v1.1 does not beat `SoftWeight_noReject_LS010`, so the artifact gate remains
    the unstable part on ST.
  - Do not migrate this exact v1.1 policy to CBraMod as a success claim. Either
    revise the gate/weight interaction under a new locked hypothesis or park ST
    v1.1 as a negative validation.
- Primary outputs:
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/SASCERT_SOFTAR_LS_V1_1_REPORT_v1_1_source_tuned_full.md`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/compact_sascert_v1_1_result_v1_1_source_tuned_full.json`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/steegformer_physionetmi_sascert_metrics_v1_1_source_tuned_full.csv`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/steegformer_physionetmi_paired_comparison_v1_1_source_tuned_full.csv`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/certificate_score_distribution_v1_1_source_tuned_full.csv`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/rejected_samples_summary_v1_1_source_tuned_full.csv`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/failure_cases_summary_v1_1_source_tuned_full.csv`
  - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/outputs/leakage_audit_v1_1_source_tuned_full.json`
- Safety/compliance:
  - No new backbone downloads.
  - No paper reproduction.
  - No MIRepNet / EEGPT.
  - No dataset switch.
  - No raw EDF copied.
  - No old experiment outputs modified.
  - No hard Top50.
  - No target-test leakage detected.

### 2026-06-23 Workspace And Ledger Confirmation

- User asked whether the v1.1 validation was performed inside the project
  workspace and whether it was recorded in project files.
- Confirmed:
  - Current workspace:
    - `/ai/224duibishiyan/615新研究`
  - Workbench trial exists:
    - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi`
  - Trial status:
    - `completed`
    - decision `do_not_promote_v1_1_on_st`
  - Long-term ledger contains the v1.1 validation section:
    - `2026-06-23 SAS-Cert-SoftAR-LS v1.1 ST-EEGFormer PhysioNetMI Validation`
- Current no-op status:
  - This turn only confirmed paths/status and updated the ledger; no new
    experiments, training runs, downloads, code changes outside this ledger,
    data copies, paper edits, figures, thresholds, or result files were
    introduced.

### 2026-06-23 Follow-Up Code Iteration Preference

- User clarified a project-maintenance preference for upcoming prompts:
  - Reuse and iterate on existing relevant code whenever possible.
  - Avoid creating new duplicate runners, nested workbench layers, or extra
    files just because it is easier.
  - Add new files only when strictly necessary or when they clearly improve
    long-term maintainability.
- Standing rule updated accordingly.
- Practical implication:
  - Future SAS-Cert v1.1 follow-ups should prefer modifying the existing
    `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi`
    runner/config/report machinery unless a new reusable abstraction belongs in
    `sas_core/` or a stable script.
- Current no-op status:
  - This turn only updated project-maintenance rules; no experiments, training
    runs, downloads, non-ledger code changes, data copies, paper edits, figures,
    thresholds, or result files were introduced.

### 2026-06-23 SAS-Cert-SoftSafe-LS v1.2 ST-EEGFormer PhysioNetMI Repair

- Task completed:
  - `SASCERT_V1_2_SOFTSAFE_REPAIR_ON_STEEGFORMER_PHYSIONETMI`
- Code reuse:
  - No duplicate runner was created.
  - Existing runner iterated:
    - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py`
  - Added:
    - `--experiment v1_2`
    - `--output-dir`
    - SoftSafe scoring fields
    - normalized augmentation-loss training path
    - v1.1 gate-harm and loss-mass audit writers
- Workbench output trial:
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi`
- Scope:
  - Backbone: `ST-EEGFormer-small`
  - Dataset: `PhysioNetMI / EEGMMI`
  - Task: left-vs-right MI
  - Runs: `R04/R08/R12`
  - Checkpoint:
    `outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt`
  - Targets: `90-109`
  - Seeds: `20,21,22,23,24`
  - Same candidate augmentation pool, optimizer, epochs, batch size, and split
    policy as v1.1.
- Implemented v1.2 changes:
  - hard top-10 artifact reject removed from the main method.
  - extreme artifact outlier rule:
    - `robust_z(artifact_score) > 4` or non-finite artifact score gets
      `w = 0`.
  - artifact score becomes a soft safety penalty:
    - `artifact_safe = 1 - ranknorm(artifact_score)`.
  - prediction consistency retained for audit only, not final v1.2 weight.
  - content:
    - `ranknorm(E_embed) + ranknorm(E_proto)`.
  - style weight reduced from `0.5` to `0.3`.
  - final:
    - `E_base = E_content + E_physio + 0.3 * E_style`
    - `E_total = E_base * (0.5 + 0.5 * artifact_safe)`
    - `w = 0.5 + 0.5 * ranknorm(E_total)`
  - training loss:
    - `L = CE_real + sum(w_i * CE_aug_i) / (sum(w_i) + eps)`
    - label smoothing `0.10`.
- Full command:
  - `PYTHONPATH=outputs/foundation_physio_mi_fullfinetune/local_python_deps:. python workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py --experiment v1_2 --targets 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 --seeds 20 21 22 23 24 --device cuda --source-epochs 30 --finetune-epochs 80 --batch-size 64 --feature-batch-size 64 --steegformer-state-dict outputs/foundation_physio_mi_fullfinetune/checkpoints/st_eegformer_small_seed3407_best.pt --feature-tag st_source_tuned_seed3407 --output-tag v1_2 --output-dir workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs`
- Full result:
  - `NaiveAug_LS010`:
    - BAcc `0.7101`, Macro-F1 `0.7052`, AUROC `0.7810`, ECE `0.2156`,
      NLL `0.7338`, Brier `0.4264`.
  - `SoftWeight_noReject_LS010`:
    - BAcc `0.7100`, Macro-F1 `0.7049`, AUROC `0.7812`, ECE `0.2156`,
      NLL `0.7327`, Brier `0.4241`.
  - `SAS-Cert-SoftSafe-LS-v1.2`:
    - BAcc `0.7070`, Macro-F1 `0.7019`, AUROC `0.7829`, ECE `0.2135`,
      NLL `0.7340`, Brier `0.4251`.
- Primary comparisons:
  - v1.2 vs Naive:
    - Delta BAcc `-0.0030`
    - Delta Macro-F1 `-0.0032`
    - Delta ECE `-0.0021`
    - Delta NLL `+0.0002`
    - Delta Brier `-0.0014`
  - v1.2 vs SoftWeight no-reject:
    - Delta BAcc `-0.0030`
    - Delta Macro-F1 `-0.0030`
    - Delta ECE `-0.0021`
    - Delta NLL `+0.0013`
    - Delta Brier `+0.0010`
  - Subject win rate Macro-F1 vs Naive: `0.15`
  - Seed win rate Macro-F1 vs Naive: `0.00`
- v1.1 failure audit:
  - v1.1 rejected candidates had higher average content score:
    - rejected `0.5404`
    - kept `0.4955`
  - v1.1 rejected candidates also had higher average physio score:
    - rejected `0.5451`
    - kept `0.4950`
  - Interpretation:
    - v1.1 hard gate did reject relatively high-content/high-physio candidates.
- Loss mass audit:
  - v1.1 SoftAR effective augmentation scale: `0.5351`
  - v1.2 SoftSafe effective augmentation scale: `0.6868`
  - v1.2 repaired the low weight-mass issue, but the repaired loss mass did not
    recover ST classification utility.
- v1.2 certificate summary:
  - rejected ratio: `0.0965`
  - mean weight: `0.6868`
  - sum weight / candidate: `0.6868`
- Leakage audit:
  - `passed`
  - target test was not used for thresholds, ranknorm, prototype, style anchor,
    best epoch, or best seed.
- Decision:
  - `ARTIFACT_PHYSIO_STYLE_NOT_HELPING_ST`
  - Do not enter CBraMod replication with this current SoftSafe policy.
  - Recommended next branch:
    - retreat to content-only soft weighting, or archive artifact/physio/style
      weighting as diagnostic-only for ST.
- Required outputs:
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/SASCERT_SOFTSAFE_V1_2_REPORT.md`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/compact_sascert_v1_2_result.json`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/metrics_v1_2.csv`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/paired_comparison_v1_2.csv`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/gate_harm_audit.csv`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/loss_mass_audit.csv`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/certificate_distribution_v1_2.csv`
  - `workbench/20260623_sascert_softsafe_v1_2_steegformer_physionetmi/outputs/leakage_audit_v1_2.json`
- Safety/compliance:
  - No new tool downloads.
  - No backbone switch.
  - No dataset switch.
  - No MIRepNet / EEGPT.
  - No paper reproduction.
  - No hard Top50.
  - No target-test leakage detected.

### 2026-06-23 Project Intermediate/Workbench System Clarification

- User asked to locate the earlier "中转系统" for preventing the project from
  accumulating uncontrolled one-off scripts and project folders.
- Clarification:
  - There is no separate tool literally named `中转系统`.
  - The implemented project-level intermediate system is the `workbench/` trial
    box workflow.
  - This is different from a download relay/proxy/mirror system; the previous
    "relay clarification" referred only to downloads.
- Core files:
  - `docs/PROJECT_WORKFLOW.md`
  - `workbench/README.md`
  - `scripts/manage_trial.py`
  - `PROJECT_INDEX.md`
  - `scripts/SCRIPT_REGISTRY.md`
  - `RUN_REGISTRY.md`
  - `PATCHES.md`
- Workflow:
  - New ideas start in `workbench/<date>_<trial_name>/`.
  - Create a trial with `python scripts/manage_trial.py new <trial_name>`.
  - Each trial contains `TRIAL.md`, `config.yaml`, `status.json`, and
    `outputs/`.
  - Successful reusable logic is promoted into `sas_core/`, stable `scripts/`,
    or `configs/experiments/`.
  - Failed trials are summarized and archived/parked instead of being mixed into
    the stable code path.
- Existing examples:
  - `workbench/20260622_steegformer_sascert_core`
  - `workbench/20260622_cbramod_physionetmi_sascert_matched`
  - `workbench/20260622_cross_backbone_cert_direction_audit`
  - `workbench/20260622_component_gated_cert_rule`
  - `workbench/20260623_reference_algorithm_audit`

### 2026-06-24 GitHub Upload Boundary and Directory Cleanup Audit

- User asked to connect the project to GitHub, upload only appropriate project
  files, and organize stale project directories.
- Current Git state:
  - Repository has no commits yet on `master`.
  - No Git remote is configured.
  - `gh` GitHub CLI is not installed in the environment, so no authenticated
    GitHub push or PR could be performed in this pass.
- Upload boundary tightened:
  - `.gitignore` was updated to exclude generated experiment outputs,
    nested `third_party/` downloads, LaTeX build artifacts, runtime caches, and
    large/raw EEG or checkpoint-like binary formats.
  - Candidate unignored upload set is now about `254` files / `4.1M`, reduced
    from the full workspace size of about `9.0G`.
- Recommended files to upload:
  - Project governance docs: `README.md`, `PROJECT_INDEX.md`,
    `PROJECT_MANAGEMENT.md`, `RUN_REGISTRY.md`, `PATCHES.md`, `.gitignore`.
  - Stable reusable code: `sas_core/`, selected stable `scripts/`, and
    `configs/experiments/`.
  - Workbench metadata and source scripts only: `TRIAL.md`, `config.yaml`,
    `status.json`, and trial scripts; do not upload `workbench/*/outputs/`.
  - Paper source files and figure/table sources under `paper/`; keep generated
    LaTeX build intermediates out of Git.
  - Lightweight documentation under `docs/` and README placeholders that define
    ignored artifact/output locations.
- Recommended files not to upload:
  - `outputs/**` except README placeholders.
  - `third_party/**` and nested downloaded external repositories.
  - Raw data, cached arrays, model weights, checkpoints, archives, and LMDB/MDB
    files.
  - Historical generated outputs such as `sas_cert_mve_outputs/**` and
    `sas_cert_cbramod_mve/outputs/**`.
  - Workbench generated result CSV/JSON/Markdown outputs unless promoted into a
    stable report.
  - Local utility/runtime state such as GPU memory hold logs.
- Needs user decision before staging:
  - Whether personal/local notes `参考论文集/`, `复盘助手`, and `选题.txt` should
    be public, moved into a private notes area, or ignored.
  - Whether `显存占位/` GPU helper scripts should be kept in the public repo or
    treated as local machine utilities.
- No files were deleted or moved during this audit.

### 2026-06-24 GitHub Initial Upload Pause Point

- Target repository supplied by user:
  - `https://github.com/doukai589/research2.git`
- Local repository preparation completed:
  - `origin` remote was configured to the target repository.
  - GitHub CLI `gh` was installed locally (`2.95.0`).
  - Branch was normalized to `main`.
  - Curated upload set contains `245` tracked files, about `3.1M`.
  - Excluded from Git:
    - large/generated outputs
    - downloaded third-party repositories
    - raw EEG/data/cache/checkpoint/archive binaries
    - workbench output directories
    - personal/local notes: `参考论文集/`, `复盘助手`, `选题.txt`
    - local GPU utility directory: `显存占位/`
- Local commits currently present:
  - `30f5860` `Initial curated research project upload`
  - `b647160` `first commit`
- Upload status:
  - Push to GitHub was not completed because the environment is not yet
    authenticated for the private GitHub repository.
  - `gh auth status` reported no logged-in GitHub host.
  - `git push -u origin main` failed on missing HTTPS credentials.
- User asked to pause and verify personally before continuing.
- Resume point:
  - After the user completes GitHub authentication or confirms the remote state,
    continue from local `main`, verify `git status -sb`, then push with
    `git push -u origin main` if the remote is ready.

### 2026-06-24 Manual GitHub Upload and Verification Instructions

- User asked for manual upload commands and verification steps.
- Current local state before user-side verification:
  - Branch: `main`
  - Remote: `origin -> https://github.com/doukai589/research2.git`
  - Existing commits:
    - `30f5860` curated project upload
    - `b647160` small README follow-up
  - `PROJECT_MANAGEMENT.md` has the newest GitHub pause/instruction ledger
    changes and should be committed before the user's manual push.
- Recommended manual flow:
  - authenticate GitHub with `gh auth login`
  - commit the latest `PROJECT_MANAGEMENT.md` ledger change
  - push `main` to `origin`
  - verify with `git status -sb`, `git ls-remote origin refs/heads/main`, and
    the GitHub repository page.

### 2026-06-24 GitHub Tracking Policy Revision: Code plus Lightweight Outputs

- User clarified the intended GitHub purpose:
  - GitHub should record what new code was written.
  - GitHub should also record the outputs produced by each run, so future review
    can see what happened in each task.
  - Assistant final replies should explicitly state what was done and which new
    files/outputs were produced.
- Policy revision:
  - Track new or modified code/config/documentation files.
  - Track lightweight experiment outputs:
    - `*.md`
    - `*.csv`
    - `*.json`
    - `*.txt`
    - `*.yaml`
    - `*.yml`
  - Continue excluding heavyweight or non-reviewable artifacts:
    - raw EEG/data files
    - cached arrays and feature tensors: `*.npy`, `*.npz`
    - model/checkpoint files: `*.pt`, `*.pth`, `*.ckpt`
    - archives and compressed downloads
    - downloaded third-party repositories
    - dependency snapshots inside output folders, such as `local_python_deps/`
      and `python_target/`
    - local/private notes and machine utilities
- `.gitignore` was updated accordingly:
  - `outputs/**` lightweight result files are now upload candidates.
  - `workbench/*/outputs/**` lightweight result files are now upload
    candidates.
  - `sas_cert_mve_outputs/**` lightweight result files are now upload
    candidates.
  - `sas_cert_cbramod_mve/outputs/**` lightweight result files are now upload
    candidates.
- Current effect of the revised rule:
  - About `1145` currently untracked lightweight output files become Git
    candidates.
  - Total candidate size is about `27M`.
  - No raw data, checkpoints, downloaded third-party repositories, dependency
    snapshots, or binary experiment caches are included.
- Required answer/reporting habit going forward:
  - Every final task response should include:
    - what was done
    - code/config/doc files changed
    - output files created or updated
    - whether files are ready for GitHub upload
    - any files intentionally not uploaded and why

### 2026-06-24 GitHub Upload Status Check

- User asked to check whether the GitHub upload/setup is now OK.
- Local check results:
  - Current branch: `main`
  - Local branch tracks `origin/main`.
  - Latest tracked commit:
    - `b647160` `first commit`
  - Previous curated project upload commit:
    - `30f5860` `Initial curated research project upload`
- Interpretation:
  - The first curated upload appears to be synchronized to `origin/main`.
  - The newer GitHub tracking-policy update is not yet committed or pushed.
- Current uncommitted changes:
  - `.gitignore`
    - changed to allow lightweight outputs (`md/csv/json/txt/yaml/yml`) while
      keeping heavy/raw/cache/dependency files ignored.
  - `PROJECT_MANAGEMENT.md`
    - includes GitHub pause point, manual upload instructions, tracking-policy
      revision, and this status check.
- Current untracked Git candidates under the revised policy:
  - about `1145` lightweight output files
  - total size about `27M`
  - no raw data, checkpoints, downloaded third-party repositories, dependency
    snapshots, or binary experiment caches detected in the candidate set.
- GitHub authentication status from this environment:
  - `gh auth status` still reports no logged-in GitHub host.
  - GitHub connector could not read the private repository content.
- Verdict:
  - Initial project upload: likely OK.
  - Full desired state including lightweight outputs and new tracking rules:
    not yet complete until `.gitignore`, `PROJECT_MANAGEMENT.md`, and the
    lightweight output files are committed and pushed.

### 2026-06-24 GitHub Upload Verified After User Push

- User reported `git status -sb` as clean with `main...origin/main`.
- Local re-check results:
  - Latest synchronized commit:
    - `20cd6ad` `Track lightweight experiment outputs`
  - Branch:
    - `main`
  - Tracking:
    - `main` tracks `origin/main`
- Interpretation:
  - The curated project upload has been pushed.
  - The lightweight experiment outputs and revised tracking policy have also
    been pushed.
- Remaining local-only file:
  - `github上传命令`
  - This is a temporary command note, not a project source/output artifact.
  - It was added to `.gitignore` instead of being deleted or uploaded.
- Current GitHub policy remains:
  - upload code/config/docs
  - upload lightweight output files (`md/csv/json/txt/yaml/yml`)
  - do not upload raw data, checkpoints, large caches, third-party downloads,
    dependency snapshots, or local personal notes

### 2026-06-24 GitHub Web/Keyword Verification

- User asked to visit `https://github.com/doukai589/research2` and search key
  terms with GitHub code search.
- GitHub repository web page:
  - Accessible.
  - Repository is visible as `doukai589/research2`.
  - Page shows `Public`, branch `main`, `4 Commits`, and expected project
    directories including `outputs`, `workbench`, `sas_core`, `scripts`,
    `docs`, `paper`, `sas_cert_mve_outputs`, and `sas_cert_cbramod_mve`.
- GitHub code search:
  - Direct GitHub code search URL returned HTTP `429 Too Many Requests`.
  - Because web code search was rate-limited, keyword verification was performed
    against the synchronized local checkout.
- Local sync status during verification:
  - `git status -sb`: clean, `main...origin/main`.
  - `HEAD` and `origin/main`: `00894f8 Record GitHub upload verification`.
  - `git ls-remote` hit a transient TLS termination error, but local branch
    tracking and repository web page both indicate the repository is uploaded.
- Keyword hit counts in the synchronized checkout:
  - `sascert`: `2605`
  - `SAS-Cert`: `1415`
  - `SoftAR`: `2747`
  - `SoftSafe`: `877`
  - `ST-EEGFormer`: `628`
  - `PhysioNetMI`: `785`
  - `pyriemann`: `255`
  - `mne_features`: `216`
  - `artifact_score`: `20`
  - `E_content`: `1134`
  - `E_physio`: `363`
  - `ranknorm`: `307`
  - `label_smoothing`: `82`
  - `v1_1`: `322`
  - `v1_2`: `385`
- Conclusion:
  - GitHub upload is visible from the web.
  - The uploaded project contains the expected SAS-Cert, SoftAR/SoftSafe,
    ST-EEGFormer, PhysioNetMI, pyRiemann, MNE-Features, and v1.1/v1.2
    references.

### 2026-06-24 Interpretation of External v1.2 Analysis

- User provided an external AI analysis of the v1.2 SoftSafe result and asked
  only to understand it.
- No experiment code was changed and no new experiment was launched.
- Key absorbed interpretation:
  - v1.2 is not a successful reliable augmentation method.
  - v1.2 is still scientifically useful because it confirms the failure
    diagnosis:
    - v1.1 hard artifact gate rejected samples with higher mean content and
      physio scores than kept samples.
    - v1.2 repaired part of the low augmentation loss-mass problem.
    - v1.2 still did not recover classification utility on ST-EEGFormer-small +
      PhysioNetMI.
  - Artifact, physio, and style scores are useful as diagnostic certificate
    components, but current evidence does not support directly using them as
    ST training weights.
- Important conceptual split to preserve:
  - Diagnostic Certificate:
    - content stability
    - artifact risk
    - physiological fidelity
    - style plausibility
    - prediction consistency
  - Utility Policy:
    - should be decoupled from the full diagnostic certificate
    - next ST training utility should be content-only unless audit evidence says
      otherwise
- Proposed next branch from the external analysis:
  - `SAS-Cert-CU-LS v1.3`
  - CU = Content Utility
  - diagnosis remains multidimensional
  - training weight uses only:
    - `E_embed`
    - `E_proto`
    - `E_content = ranknorm(E_embed) + ranknorm(E_proto)`
  - exclude from main training weight:
    - `artifact_score`
    - `E_physio`
    - `E_style`
    - `E_pred`
  - suggested weight:
    - `q = ranknorm(E_content)`
    - `w = 0.75 + 0.5 * q`
    - weight range `[0.75, 1.25]`
  - preserve normalized augmented loss:
    - `L_aug = sum(w_i * CE_i) / (sum(w_i) + eps)`
    - `L = CE_real + L_aug`
    - label smoothing `0.10`
- Recommended before v1.3 experiment:
  - run a Component Utility Audit over existing v1.1/v1.2 cached scores.
  - test correlations between score components and:
    - raw CE loss
    - augmented-sample correctness
    - augmentation type
    - subject
    - label
  - use Spearman correlation and top/bottom score comparisons.
- CBraMod replication should wait:
  - enter CBraMod only if v1.3 is at least not worse on ST, or shows a clear
    reliability/calibration trade-off.
  - if v1.3 fails on ST, mark ST branch as diagnostic-only or move to a
    risk-mixed candidate pool experiment.
- Working conclusion:
  - Do not continue searching for more tools at this stage.
  - The next decision point is algorithmic: whether content-only utility can
    provide training benefit while the full SAS-Cert remains diagnostic.

### 2026-06-24 SAS-Cert-CU-LS v1.3 ST-EEGFormer PhysioNetMI

- Task:
  - `SASCERT_V1_3_CONTENT_UTILITY_REPAIR_ON_STEEGFORMER_PHYSIONETMI`
- Goal:
  - decouple multidimensional diagnostic certificate from training utility.
  - keep artifact / physio / style / prediction consistency as diagnostics.
  - use content-only utility for training weight.
- Implementation:
  - Reused and extended existing runner:
    - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py`
  - Added `v1_3` experiment branch.
  - Added `SAS-Cert-CU-LS-v1.3` group.
  - Added candidate-level raw CE loss and correctness audit from augmented
    training candidates only.
  - Added content-only training weight:
    - `E_content = ranknorm(E_embed) + ranknorm(E_proto)`
    - `w = 0.75 + 0.5 * ranknorm(E_content)`
  - Preserved normalized augmented loss:
    - `L_aug = sum(w_i * CE_i) / (sum(w_i) + eps)`
    - `L = CE_real + L_aug`
    - label smoothing `0.10`
  - No artifact hard gate was used for v1.3.
  - NaN/Inf content samples are skipped only through zero weight; observed
    skipped count was `0`.
- Trial directory:
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi`
- Full run:
  - Backbone: ST-EEGFormer-small
  - Dataset: PhysioNetMI / EEGMMI left-vs-right MI
  - Runs: R04/R08/R12
  - Targets: `90-109`
  - Seeds: `20-24`
  - Groups:
    - `NaiveAug_LS010`
    - `SoftWeight_noReject_LS010`
    - `SAS-Cert-CU-LS-v1.3`
- Component Utility Audit:
  - Training utility candidates:
    - `E_embed`
    - `E_proto`
    - `E_content`
  - Diagnostic-only or unstable for training utility:
    - `E_pred`
    - `artifact_score`
    - `artifact_safe`
    - `E_physio`
    - `E_style`
    - `D_band`
    - `D_cov`
    - `D_style`
  - Strongest low-CE / high-correctness signals:
    - `E_proto`: Spearman CE `-0.8877`, correctness `+0.4623`
    - `E_content`: Spearman CE `-0.5293`, correctness `+0.2972`
    - `E_embed`: Spearman CE `-0.0892`, correctness `+0.0503`
- Main results:
  - `NaiveAug_LS010`:
    - BAcc `0.7567`
    - Macro-F1 `0.7524`
    - ECE `0.1857`
    - NLL `0.6221`
    - Brier `0.3594`
  - `SoftWeight_noReject_LS010`:
    - BAcc `0.7568`
    - Macro-F1 `0.7527`
    - ECE `0.1826`
    - NLL `0.6150`
    - Brier `0.3559`
  - `SAS-Cert-CU-LS-v1.3`:
    - BAcc `0.7588`
    - Macro-F1 `0.7543`
    - ECE `0.1861`
    - NLL `0.6206`
    - Brier `0.3587`
- Required comparisons:
  - v1.3 vs `NaiveAug_LS010`:
    - delta BAcc `+0.002078`
    - delta Macro-F1 `+0.001935`
    - delta ECE `+0.000444`
    - delta NLL `-0.001490`
    - delta Brier `-0.000763`
  - v1.3 vs `SoftWeight_noReject_LS010`:
    - delta BAcc `+0.001959`
    - delta Macro-F1 `+0.001634`
    - delta ECE `+0.003503`
    - delta NLL `+0.005643`
    - delta Brier `+0.002736`
  - Subject win rate Macro-F1 vs Naive:
    - `0.05`
  - Seed win rate Macro-F1 vs Naive:
    - `0.00`
  - Mean weight:
    - `1.000000`
  - Weight range:
    - `[0.75, 1.25]`
- Leakage audit:
  - `passed`
  - target test was not used for thresholds, ranknorm, prototype, style anchor,
    component utility audit, best epoch, or best seed.
- Decision:
  - `enter_cbramod_recheck`
  - Rationale:
    - v1.3 improves BAcc and Macro-F1 vs both NaiveAug and SoftWeight.
    - v1.3 improves NLL/Brier vs Naive with only tiny ECE increase.
    - v1.3 has a small calibration trade-off vs SoftWeight, so CBraMod should
      be treated as a recheck rather than broad promotion.
- Required outputs:
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/SASCERT_CU_V1_3_REPORT.md`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/compact_sascert_v1_3_result.json`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/metrics_v1_3.csv`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/paired_comparison_v1_3.csv`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/component_utility_audit.csv`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/component_utility_summary.json`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/diagnostic_scores_v1_3.csv`
  - `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/leakage_audit_v1_3.json`
- GitHub tracking:
  - Code/config/doc changes and lightweight outputs should be committed.
  - Heavy feature cache remains ignored:
    - `outputs/features/original_st_features_st_source_tuned_seed3407.npz`

### 2026-06-24 Explanation of v1.3 Algorithm Improvement Effect

- User asked for a detailed explanation of the effect after the v1.3 algorithm
  improvement.
- Interpretation recorded:
  - v1.3 is the first ST-EEGFormer-small + PhysioNetMI SAS-Cert training branch
    that improves both BAcc and Macro-F1 over `NaiveAug_LS010`.
  - v1.3 also improves BAcc and Macro-F1 over `SoftWeight_noReject_LS010`, but
    with a small calibration/probability-quality trade-off compared with
    SoftWeight:
    - ECE `+0.003503`
    - NLL `+0.005643`
    - Brier `+0.002736`
  - The main algorithmic lesson is that the diagnostic certificate and training
    utility policy should be decoupled.
  - `E_proto` and `E_content` provide strong evidence for content utility:
    - `E_proto` Spearman CE `-0.8877`, correctness `+0.4623`
    - `E_content` Spearman CE `-0.5293`, correctness `+0.2972`
  - `artifact_score`, `E_physio`, and `E_style` did not show enough direct
    training-utility signal for ST and should remain diagnostic-only.
  - The effect is a real but modest repair:
    - classification direction is positive.
    - Naive calibration is not harmed materially.
    - subject/seed reliability remains weak, so this is not yet a fully robust
      method.
- Practical conclusion:
  - Proceed to CBraMod recheck as a validation step.
  - Do not broadly promote v1.3 as final until cross-backbone behavior and
    reliability are checked.

### 2026-06-24 Workbench Relay and Project Management Reminder

- User asked whether the project relay/intermediate system and project
  management rules are remembered.
- Confirmed:
  - The "中转站" is the `workbench/` trial-box workflow.
  - New exploratory experiments should start in `workbench/<date>_<trial_name>/`.
  - Each trial should keep `TRIAL.md`, `config.yaml`, `status.json`, and
    lightweight `outputs/`.
  - Reusable stable code should be promoted back into `sas_core/`, stable
    `scripts/`, or `configs/experiments/` only after validation.
  - `PROJECT_MANAGEMENT.md` remains the long-term project ledger and should be
    updated after each answered request.
  - Final responses should state what changed, what outputs were produced, and
    what should be uploaded to GitHub.

### 2026-06-24 SAS-Cert v1.4 SCB-CU Risk-Mixed Stress Test

- Task:
  - `SASCERT_V1_4_SUBJECT_CLASS_BALANCED_CU_AND_RISK_MIXED_STRESS_TEST`
- Workbench trial:
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi`
- Code policy:
  - Reused and extended existing runner:
    - `workbench/20260623_sascert_softar_ls_v1_1_steegformer_physionetmi/runner_v1_1.py`
  - Did not download new tools.
  - Did not switch backbone or dataset.
  - Did not add artifact / physio / style / prediction consistency back into
    training weights.
- Implemented:
  - v1.4 `SAS-Cert-SCB-CU-LS-v1.4`.
  - Subject-class ranknorm over `E_content = ranknorm(E_embed) + ranknorm(E_proto)`.
  - Weight rule:
    - `w = 0.75 + 0.5 * ranknorm_subject_class(E_content)`
  - Class-balanced weighted augmentation loss:
    - `L_aug = mean_c sum(w_i * CE_i) / (sum(w_i) + eps)`
    - `L = CE_real + L_aug`
  - Risk-mixed candidate pool:
    - 70% mild augmentations.
    - 30% risky augmentations.
    - risky types include strong frequency mask, strong channel dropout,
      EMG-like burst, EOG-like drift, and covariance perturbation.
- v1.3 localization audit:
  - `E_proto` remains strongest:
    - mean Spearman CE `-0.8503`
    - mean Spearman correctness `+0.4572`
  - `E_content` remains useful:
    - mean Spearman CE `-0.5669`
    - mean Spearman correctness `+0.3305`
  - `E_embed` is weak:
    - mean Spearman CE `-0.0748`
    - mean Spearman correctness `+0.0794`
  - Subject/class balancing removed average weight unfairness:
    - all subject/class bins have mean v1.4 weight `1.0`
    - ranknorm scope used: `subject_class`
  - Subject win stability remained weak, so global ranknorm unfairness is not
    the main failure mode.
- Regular pool results:
  - v1.4 vs `NaiveAug_LS010`:
    - delta BAcc `+0.000626`
    - delta Macro-F1 `+0.000295`
    - delta ECE `-0.001412`
    - delta NLL `-0.001908`
    - delta Brier `-0.000977`
  - v1.4 vs v1.3:
    - delta BAcc `-0.001453`
    - delta Macro-F1 `-0.001640`
    - subject win rate Macro-F1 `0.00`
- Risk-mixed pool results:
  - v1.4 vs `RiskMixed_NaiveAug_LS010`:
    - delta BAcc `+0.000476`
    - delta Macro-F1 `+0.000450`
    - delta ECE `-0.001121`
    - delta NLL `-0.003801`
    - delta Brier `-0.002236`
    - subject win rate Macro-F1 `0.00`
    - seed win rate Macro-F1 `0.00`
- Leakage audit:
  - `passed`
  - target test was not used for ranknorm, prototype, threshold, risk-mixed
    pool construction, best epoch, or best seed.
- Decision:
  - `limit_training_use_to_diagnostic_or_riskmixed`
  - v1.4 does not justify CBraMod recheck.
  - If continuing the training branch, focus on subject-balanced utility repair
    or explicitly risk-mixed augmentation settings rather than treating v1.4 as
    a general ST improvement.
- Required outputs:
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/SASCERT_V1_4_SCB_CU_RISKMIXED_REPORT.md`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/compact_sascert_v1_4_result.json`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/metrics_v1_4_regular_pool.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/paired_comparison_v1_4_regular_pool.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/metrics_v1_4_riskmixed_pool.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/paired_comparison_v1_4_riskmixed_pool.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/per_subject_delta_table.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/per_class_delta_table.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/per_subject_component_corr.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/weight_distribution_by_subject_class.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/riskmixed_diagnostic_summary.csv`
  - `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/leakage_audit_v1_4.json`
- GitHub tracking:
  - Commit code/config/docs and lightweight v1.4 outputs.
  - No raw EEG, checkpoints, feature caches, or third-party dependency trees
    should be uploaded.
  - Local commit:
    - `3e1383f Run SAS-Cert SCB-CU v1.4 ST PhysioNetMI`
  - Push status:
    - blocked locally by missing HTTPS GitHub credentials:
      `could not read Username for 'https://github.com'`
