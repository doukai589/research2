# MIRepNet Code Entry Audit

- root: `/ai/224duibishiyan/615新研究/third_party/backbones/MIRepNet`
- finetune.py: `True`
- dataset.py: `True`
- model dir: `True`
- weight: `True`

## Official Support Signals
- README_BNCI2014004: `True`
- README_BNCI2014001: `False`
- dataset_py_BNCI2014001_4: `True`
- expects_X_npy: `True`
- expects_labels_npy: `True`

dataset.py expects ./data/<dataset>/X.npy and labels.npy. PatchEmbedding defaults to 45 channels. README documents BNCI2014004 command and says code for other datasets will be released.
