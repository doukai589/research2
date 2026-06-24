# CBraMod Code Entry Audit

- official_root: `/ai/224duibishiyan/615新研究/third_party/CBraMod-main`
- MVE model/root with weights: `/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve/third_party/CBraMod`
- finetune_main.py exists: `True`
- finetune_trainer.py exists: `True`
- datasets exists: `True`
- preprocessing exists: `True`
- requirements exists: `True`

Official downstream code consumes processed LMDB datasets, not raw MAT/EDF directly. The dry run therefore creates a tiny LMDB sample in the repair output directory; it does not copy raw datasets.
