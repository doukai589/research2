# Run Registry

This file indexes durable runs and their primary output paths.

## Active / Important Runs

| Run | Protocol | Primary Output | Status |
|---|---|---|---|
| CBraMod PhysioNetMI paper-code | `code_exact_on_existing_paper_aligned_lmdb` | `outputs/paper_code_runs/PAPER_CODE_RUN_REPORT.md` | completed |
| Foundation PhysioNetMI full fine-tune | `paper_aligned_common_protocol` | `outputs/foundation_physio_mi_fullfinetune/PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md` | completed |
| SAS-Cert v2 ST PhysioNetMI | `structured certificate + CertAdapter` | `workbench/20260624_sascert_v2_certadapter_steegformer_physionetmi/outputs/SASCERT_V2_REPORT.md` | completed |
| SAS-Cert-SCB-CU-LS v1.4 ST PhysioNetMI | `subject-class balanced CU + risk-mixed stress test` | `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs/SASCERT_V1_4_SCB_CU_RISKMIXED_REPORT.md` | completed |
| SAS-Cert-CU-LS v1.3 ST PhysioNetMI | `ST-EEGFormer-small + PhysioNetMI content-only utility` | `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs/SASCERT_CU_V1_3_REPORT.md` | completed |
| SAS-Cert-CBraMod MVE | BCIC-IV-2a few-shot SAS-Cert | `sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_V5_LOCKED_CONFIRMATORY_REPORT.md` | completed |
| MIRepNet reproduction | BNCI2014001-4 / BNCI2014001 adapter attempts | `outputs/mirepnet_full_paper_code/MIRepNet_MOABB_SESSION_T_RERUN_REPORT.md` | parked |

## Historical Output Roots

- `outputs/setup_audit`
- `outputs/setup_audit_step1`
- `outputs/repro_gate_repair`
- `outputs/paper_reproduction`
- `outputs/paper_code_runs`
- `outputs/mirepnet_full_paper_code`
- `outputs/foundation_physio_mi_fullfinetune`
- `workbench/20260624_sascert_v2_certadapter_steegformer_physionetmi/outputs`
- `workbench/20260623_sascert_v1_4_scb_cu_riskmixed_steegformer_physionetmi/outputs`
- `workbench/20260623_sascert_cu_v1_3_steegformer_physionetmi/outputs`
- `sas_cert_mve_outputs`
- `sas_cert_cbramod_mve/outputs`
