# SAS-Cert-SoftSafe-LS v1.2 ST-EEGFormer PhysioNetMI Report

## Main Result

| Group | BAcc | Macro-F1 | AUROC | ECE | NLL | Brier | Rejected ratio | Mean weight | Sum weight / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NaiveAug_LS010 | 0.7101 | 0.7052 | 0.7810 | 0.2156 | 0.7338 | 0.4264 | 0.0000 | 1.0000 | 1.0000 |
| SoftWeight_noReject_LS010 | 0.7100 | 0.7049 | 0.7812 | 0.2156 | 0.7327 | 0.4241 | 0.0000 | 0.6000 | 0.6000 |
| SAS-Cert-SoftSafe-LS-v1.2 | 0.7070 | 0.7019 | 0.7829 | 0.2135 | 0.7340 | 0.4251 | 0.0965 | 0.6868 | 0.6868 |

## Required Answers

- v1.1 gate harm: rejected content mean `0.540395`, kept content mean `0.495512`.
- v1.1 loss mass: SoftAR mean weight `0.535085`, effective scale `0.535085`.
- v1.2 vs Naive: delta BAcc `-0.003036`, delta Macro-F1 `-0.003232`, delta ECE `-0.002084`, delta NLL `0.000152`, delta Brier `-0.001373`.
- v1.2 vs SoftWeight: delta BAcc `-0.002974`, delta Macro-F1 `-0.003002`, delta ECE `-0.002091`, delta NLL `0.001288`, delta Brier `0.000987`.
- Subject win rate Macro-F1 vs Naive: `0.150000`.
- Seed win rate Macro-F1 vs Naive: `0.000000`.
- Rejected ratio / mean weight / sum weight per candidate: `0.096500` / `0.686770` / `0.686770`.
- Target test leakage: `not detected`.
- Decision: `ARTIFACT_PHYSIO_STYLE_NOT_HELPING_ST`.
