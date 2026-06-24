# SAS-Cert-CBraMod MVE Final Validation Report

## 1. 完成状态

- 状态：completed
- 协议：BCIC-IV-2a LOSO few-shot，subjects=1..9，seeds=20/21/22，target T 每类 5-shot，target E 仅最终评估。
- Backbone：frozen CBraMod encoder + classifier head；未训练 CBraMod backbone。
- 数据：只引用原始 BCIC2a 路径，新项目内 `.mat` 文件数为 0。
- 协议泄漏：未发现。

## 2. 主 MVE 结果

| Layer | 判定 | 关键结果 |
| --- | --- | --- |
| Layer 0 smoke | PASS | raw [22,800]；CBraMod input [22,4,200]；pooled feature [200]；unexpected checkpoint keys=0 |
| Layer 1 bad augmentation probe | FAIL | BadContent vs Clean Acc -0.0638，Kappa -0.0850；BadArtifact Acc -0.0054 |
| Layer 2 cert sorting | FAIL | overall AUC=0.6217；BadArtifact AUC=0.9474；BadContent AUC=0.8009；BadPhysio AUC=0.1595 |
| Layer 3 hard Top50 A/B | FAIL | Top50-Naive Acc -0.0237；Top50-Random50 Acc +0.0091；Top50-Naive ECE -0.0139 |

主 MVE 解释级别为 DIAGNOSTIC_ONLY。原因是 Layer 2 overall AUC 未达到 0.70，且 hard Top50 在 Layer 3 中没有超过 NaiveAug。

## 3. Shadow 验证：SoftWeight / ArtifactReject

| Group | Acc | Macro-F1 | ECE | Acc vs Naive | F1 vs Naive | ECE vs Naive | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ArtifactReject | 0.3265 | 0.2675 | 0.0744 | +0.0168 | +0.0332 | +0.0170 | FAIL |
| SASCertSoftWeight | 0.3175 | 0.2356 | 0.0648 | +0.0078 | +0.0013 | +0.0074 | PASS |
| SoftWeightArtifactReject | 0.3212 | 0.2486 | 0.0657 | +0.0116 | +0.0143 | +0.0083 | PASS |

- 最高 accuracy/F1 增益：ArtifactReject，Acc vs Naive +0.0168，Macro-F1 vs Naive +0.0332，但 ECE vs Naive +0.0170，超过 +0.01 门槛。
- 最佳可升级组：SoftWeightArtifactReject，Acc vs Naive +0.0116，Macro-F1 vs Naive +0.0143，ECE vs Naive +0.0083，通过 shadow gate。
- 纯 SoftWeight 也通过 gate：Acc vs Naive +0.0078，ECE vs Naive +0.0074。

## 4. 协议审计

| 检查项 | 结果 |
| --- | --- |
| data_copy_detected | False |
| old_outputs_used_as_input | False |
| target_test_used_for_style_anchor | False |
| target_test_used_for_cert_threshold | False |
| target_test_used_for_rank_normalization | False |
| backbone_frozen | True |
| protocol_leakage_detected | False |

## 5. 结论

1. “坏增强是真问题”在 CBraMod 版中得到部分验证：BadContent 明显伤害性能；BadArtifact 在证书排序中非常可分，但在当前训练探针中伤害幅度不强。
2. CBraMod 证书不是整体过关：overall AUC=0.6217，未达到 0.70；BadPhysio 分数方向/定义仍需重构。
3. Hard Top50 不应作为下一版主方案：它比 Random50 好一点，但比 NaiveAug 差。
4. Shadow 验证显示 soft weighting 是正确下一步：SoftWeightArtifactReject 同时提升 Acc/Macro-F1，并把 ECE 变差控制在门槛内。

## 6. 最终决策

- final_decision：PROMOTE_SOFTWEIGHT_ARTIFACTREJECT_TO_MAIN_CONFIRMATORY_MVE
- recommended_next：把 SoftWeightArtifactReject 预注册为下一轮主 A/B 组；ArtifactReject 单独保留为诊断组；在扩展模型或数据集前，优先修正 physio score。

## 7. 关键输出

- main_report: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/SAS_CERT_CBRAMOD_MVE_REPORT.md`
- main_compact_result: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/compact_result.json`
- shadow_report: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/softweight_artifactreject/SOFTWEIGHT_ARTIFACTREJECT_REPORT.md`
- shadow_summary: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/softweight_artifactreject/shadow_summary.json`
- final_report: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/FINAL_VALIDATION_REPORT.md`
- final_result: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/outputs/final_validation_result.json`

## 8. 运行命令

```bash
python scripts/run_full_mve.py --project_root /ai/224duibishiyan/615新研究/sas_cert_cbramod_mve --workspace_root /ai/224duibishiyan/615新研究 --cbramod_src ../新研究/CBraMod-main --bcic2a_root ../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014 --old_eegnet_report /ai/224duibishiyan/615新研究/sas_cert_mve_outputs/SAS_CERT_MVE_FULL_OUTPUT_AND_ANALYSIS.md --mode full --continue_on_nogo true --device auto
python scripts/07_run_softweight_artifactreject.py --project_root /ai/224duibishiyan/615新研究/sas_cert_cbramod_mve --workspace_root /ai/224duibishiyan/615新研究 --bcic2a_root ../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014 --device auto
```
