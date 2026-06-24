# Table 1. Protocol and Claim Boundary

| item | value |
| --- | --- |
| Dataset/task | PhysioNetMI / EEGMMI, left-vs-right motor imagery, runs R04/R08/R12 |
| Backbones | CBraMod frozen; ST-EEGFormer-small source-tuned frozen |
| Held-out target use | Final evaluation only; not used for score, threshold, or ranking |
| Protocol leakage audit rows | 5 |
| Supported claim boundary | supported as diagnostic certificate; directionally wrong on this pool; strongest diagnostic variant in this pack; not promoted as deployable policy; classification gain with calibration risk; positive mean but unreliable; no strong candidate-only alignment; no protocol leakage detected in audited outputs |
| Unsupported claim boundary | SAS-Cert improves few-shot accuracy reliably across subjects and seeds [do not claim]; SAS-Cert should be presented as an augmentation-selection method ready for deployment [do not claim] |
