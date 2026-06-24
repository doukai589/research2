# SAS-Cert Bibliography Resolution Report

## Verdict

```text
bibliography_placeholders_resolved_with_traceable_local_sources
```

The manuscript placeholders now have BibTeX entries in:

```text
docs/SAS_CERT_REFERENCES.bib
```

## Resolution Sources

| Key | Resolution status | Source |
|---|---|---|
| `Wang2025CBraMod` | resolved | `third_party/CBraMod-main/README.md` BibTeX |
| `Yang2026STEEGFormer` | resolved | `third_party/backbones/STEEGFormer/README.md` BibTeX |
| `Liu2026MIRepNet` | resolved | `third_party/backbones/MIRepNet/README.md` BibTeX |
| `Sirca2026LoRAEEG` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE132 |
| `Li2025FrequencyTemporalMI` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE167 |
| `Ding2025RCC` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE181 |
| `Liao2026EEGTune` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE185 |
| `Lee2025LargeBrainwaveFoundation` | resolved_from_reference_library | `参考论文集/论文集_002.txt`, RE199 |
| `Ding2026JSCCRA` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE309 |
| `Heremans2022SleepAugDomainAdapt` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE332 |
| `Sawangjai2022EEGANet` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE333 |
| `Bollens2022SubjectInvariantVAE` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE334 |
| `Winkler2011ICArtifact` | resolved_from_reference_library | `参考论文集/论文集_004.txt`, RE342 |

## Remaining Caveat

Some RE entries are resolved from the local reference-library exports rather
than publisher BibTeX. Before external submission, DOI/page metadata should be
checked for entries currently marked as `misc` or with `note` fields.
