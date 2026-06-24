# SAS-Cert-CBraMod MVE Report

## Path Audit
```json
{
  "bcic2a_mat_count": 18,
  "bcic2a_root": "/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014",
  "bcic2a_root_exists": true,
  "cbramod_src": "/ai/224duibishiyan/\u65b0\u7814\u7a76/CBraMod-main",
  "cbramod_src_exists": true,
  "cbramod_weight_exists": true,
  "data_copy_detected": false,
  "old_eegnet_report": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/sas_cert_mve_outputs/SAS_CERT_MVE_FULL_OUTPUT_AND_ANALYSIS.md",
  "old_outputs_used_as_input": false,
  "project_root": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/sas_cert_cbramod_mve",
  "workspace_exists": true,
  "workspace_root": "/ai/224duibishiyan/615\u65b0\u7814\u7a76"
}
```

## Smoke
```json
{
  "cbramod_input_shape_tail": [
    22,
    4,
    200
  ],
  "cbramod_output_shape": [
    8,
    22,
    4,
    200
  ],
  "checkpoint_audit": {
    "adapter_missing_accepted": true,
    "checkpoint_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/sas_cert_cbramod_mve/third_party/CBraMod/pretrained_weights/pretrained_weights.pth",
    "checkpoint_total_keys": 211,
    "core_missing_keys": [],
    "matched_keys": 211,
    "missing_keys": 240,
    "missing_keys_preview": [
      "encoder.layers.0.adapter.temperature",
      "encoder.layers.0.adapter.alpha",
      "encoder.layers.0.adapter.linear_adapter.down_proj.weight",
      "encoder.layers.0.adapter.linear_adapter.down_proj.bias",
      "encoder.layers.0.adapter.linear_adapter.up_proj.weight",
      "encoder.layers.0.adapter.linear_adapter.up_proj.bias",
      "encoder.layers.0.adapter.freq_adapter.spec_norm.weight",
      "encoder.layers.0.adapter.freq_adapter.spec_norm.bias",
      "encoder.layers.0.adapter.freq_adapter.mlp.0.weight",
      "encoder.layers.0.adapter.freq_adapter.mlp.0.bias",
      "encoder.layers.0.adapter.freq_adapter.mlp.3.weight",
      "encoder.layers.0.adapter.freq_adapter.mlp.3.bias",
      "encoder.layers.0.adapter.temporal_adapter.conv.weight",
      "encoder.layers.0.adapter.temporal_adapter.conv.bias",
      "encoder.layers.0.adapter.gate_linear.weight",
      "encoder.layers.0.adapter.gate_linear.bias",
      "encoder.layers.0.adapter.gate_freq.weight",
      "encoder.layers.0.adapter.gate_freq.bias",
      "encoder.layers.0.adapter.gate_time.weight",
      "encoder.layers.0.adapter.gate_time.bias",
      "encoder.layers.1.adapter.temperature",
      "encoder.layers.1.adapter.alpha",
      "encoder.layers.1.adapter.linear_adapter.down_proj.weight",
      "encoder.layers.1.adapter.linear_adapter.down_proj.bias",
      "encoder.layers.1.adapter.linear_adapter.up_proj.weight",
      "encoder.layers.1.adapter.linear_adapter.up_proj.bias",
      "encoder.layers.1.adapter.freq_adapter.spec_norm.weight",
      "encoder.layers.1.adapter.freq_adapter.spec_norm.bias",
      "encoder.layers.1.adapter.freq_adapter.mlp.0.weight",
      "encoder.layers.1.adapter.freq_adapter.mlp.0.bias"
    ],
    "model_total_keys": 451,
    "shape_mismatch": [],
    "unexpected_keys": 0
  },
  "feature_inf_count": 0,
  "feature_nan_count": 0,
  "loaded_subjects": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "loaded_trials": 5088,
  "passed": true,
  "pooled_feature_shape": [
    8,
    200
  ],
  "raw_shape_tail": [
    22,
    800
  ],
  "sessions": [
    "E",
    "T"
  ]
}
```

## Layer 1
```json
{
  "BadArtifact_delta_acc_vs_clean": -0.0054012345679012586,
  "BadArtifact_delta_kappa_vs_clean": -0.007201646090534988,
  "BadContent_delta_acc_vs_clean": -0.06378600823045272,
  "BadContent_delta_kappa_vs_clean": -0.08504801097393691,
  "BadPhysio_delta_acc_vs_clean": -0.0007716049382716084,
  "BadPhysio_delta_kappa_vs_clean": -0.0010288065843621352,
  "BadStyle_delta_acc_vs_clean": -0.009516460905349855,
  "BadStyle_delta_kappa_vs_clean": -0.012688614540466395,
  "grouped_mean": [
    {
      "acc": 0.3203446502057613,
      "condition": "NoAug",
      "ece": 0.06985357058629271,
      "kappa": 0.09379286694101509,
      "macro_f1": 0.25008678785901634,
      "n": 27,
      "nll": 1.3700497547785442
    },
    {
      "acc": 0.32304526748971196,
      "condition": "CleanAug",
      "ece": 0.06646482185211508,
      "kappa": 0.09739368998628259,
      "macro_f1": 0.2565802003380657,
      "n": 27,
      "nll": 1.3694867513797901
    },
    {
      "acc": 0.25925925925925924,
      "condition": "BadContent",
      "ece": 0.0997056531520568,
      "kappa": 0.012345679012345671,
      "macro_f1": 0.1375091095091587,
      "n": 27,
      "nll": 1.395420988400777
    },
    {
      "acc": 0.3135288065843621,
      "condition": "BadStyle",
      "ece": 0.06804220591012557,
      "kappa": 0.0847050754458162,
      "macro_f1": 0.24886632826429606,
      "n": 27,
      "nll": 1.372934897740682
    },
    {
      "acc": 0.32227366255144035,
      "condition": "BadPhysio",
      "ece": 0.07144663904874045,
      "kappa": 0.09636488340192045,
      "macro_f1": 0.25634677289672086,
      "n": 27,
      "nll": 1.3693969779544406
    },
    {
      "acc": 0.3176440329218107,
      "condition": "BadArtifact",
      "ece": 0.06756980111630856,
      "kappa": 0.0901920438957476,
      "macro_f1": 0.2511316197248461,
      "n": 27,
      "nll": 1.3675682412253485
    }
  ],
  "passed": false
}
```

## Layer 2
```json
{
  "bad_artifact_auc": 0.947394718792867,
  "bad_content_auc": 0.8008573388203017,
  "bad_physio_auc": 0.15951303155006857,
  "bad_style_auc": 0.5789948559670782,
  "bottom30_bad_rate": 0.5747599451303155,
  "overall_auc": 0.6216899862825789,
  "passed": false,
  "top30_bad_rate": 0.3628257887517147
}
```

## Layer 3
```json
{
  "paired": [
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": -0.02366255144032922,
      "metric": "acc",
      "n": 27,
      "positive_folds": 8
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": -0.044732513963169324,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 8
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": -0.013917024900807153,
      "metric": "ece",
      "n": 27,
      "positive_folds": 8
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": 0.004435570151717575,
      "metric": "nll",
      "n": 27,
      "positive_folds": 22
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": 0.00913065843621399,
      "metric": "acc",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": 0.016672778949323952,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": -0.010936497209532327,
      "metric": "ece",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": -0.005182932924341272,
      "metric": "nll",
      "n": 27,
      "positive_folds": 9
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.02160493827160494,
      "metric": "acc",
      "n": 27,
      "positive_folds": 6
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.03843432877167755,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 7
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.011644935946329016,
      "metric": "ece",
      "n": 27,
      "positive_folds": 10
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": 0.0029240360966435184,
      "metric": "nll",
      "n": 27,
      "positive_folds": 18
    }
  ],
  "summary": {
    "grouped_mean": [
      {
        "acc": 0.3203446502057613,
        "ece": 0.06985356986193071,
        "group": "NoAug",
        "kappa": 0.09379286694101509,
        "macro_f1": 0.25008678785901634,
        "n": 27,
        "nll": 1.3700497547785442
      },
      {
        "acc": 0.3224022633744856,
        "ece": 0.07212565881640885,
        "group": "NaiveAug",
        "kappa": 0.09653635116598079,
        "macro_f1": 0.2563849730505082,
        "n": 27,
        "nll": 1.36853822072347
      },
      {
        "acc": 0.28960905349794236,
        "ece": 0.06914513112513403,
        "group": "Random50",
        "kappa": 0.052812071330589856,
        "macro_f1": 0.1949796801380149,
        "n": 27,
        "nll": 1.378156723799529
      },
      {
        "acc": 0.29873971193415644,
        "ece": 0.058208633915601694,
        "group": "SASCertTop50",
        "kappa": 0.06498628257887518,
        "macro_f1": 0.21165245908733887,
        "n": 27,
        "nll": 1.3729737908751876
      }
    ],
    "paired_comparison": [
      {
        "comparison": "SASCertTop50_minus_NaiveAug",
        "mean_delta": -0.02366255144032922,
        "metric": "acc",
        "n": 27,
        "positive_folds": 8
      },
      {
        "comparison": "SASCertTop50_minus_NaiveAug",
        "mean_delta": -0.044732513963169324,
        "metric": "macro_f1",
        "n": 27,
        "positive_folds": 8
      },
      {
        "comparison": "SASCertTop50_minus_NaiveAug",
        "mean_delta": -0.013917024900807153,
        "metric": "ece",
        "n": 27,
        "positive_folds": 8
      },
      {
        "comparison": "SASCertTop50_minus_NaiveAug",
        "mean_delta": 0.004435570151717575,
        "metric": "nll",
        "n": 27,
        "positive_folds": 22
      },
      {
        "comparison": "SASCertTop50_minus_Random50",
        "mean_delta": 0.00913065843621399,
        "metric": "acc",
        "n": 27,
        "positive_folds": 13
      },
      {
        "comparison": "SASCertTop50_minus_Random50",
        "mean_delta": 0.016672778949323952,
        "metric": "macro_f1",
        "n": 27,
        "positive_folds": 14
      },
      {
        "comparison": "SASCertTop50_minus_Random50",
        "mean_delta": -0.010936497209532327,
        "metric": "ece",
        "n": 27,
        "positive_folds": 13
      },
      {
        "comparison": "SASCertTop50_minus_Random50",
        "mean_delta": -0.005182932924341272,
        "metric": "nll",
        "n": 27,
        "positive_folds": 9
      },
      {
        "comparison": "SASCertTop50_minus_NoAug",
        "mean_delta": -0.02160493827160494,
        "metric": "acc",
        "n": 27,
        "positive_folds": 6
      },
      {
        "comparison": "SASCertTop50_minus_NoAug",
        "mean_delta": -0.03843432877167755,
        "metric": "macro_f1",
        "n": 27,
        "positive_folds": 7
      },
      {
        "comparison": "SASCertTop50_minus_NoAug",
        "mean_delta": -0.011644935946329016,
        "metric": "ece",
        "n": 27,
        "positive_folds": 10
      },
      {
        "comparison": "SASCertTop50_minus_NoAug",
        "mean_delta": 0.0029240360966435184,
        "metric": "nll",
        "n": 27,
        "positive_folds": 18
      }
    ],
    "passed": false,
    "sas_top50_minus_naive_acc": -0.02366255144032922,
    "sas_top50_minus_naive_ece": -0.013917024900807153,
    "sas_top50_minus_random50_acc": 0.00913065843621399
  }
}
```

## Old EEGNet Baseline Readonly Audit
```json
{
  "old_report_exists": true,
  "old_report_path": "/ai/224duibishiyan/615\u65b0\u7814\u7a76/sas_cert_mve_outputs/SAS_CERT_MVE_FULL_OUTPUT_AND_ANALYSIS.md",
  "used_for_cache": false,
  "used_for_report_baseline_only": true,
  "used_for_threshold": false,
  "used_for_training": false
}
```

## Compact Result
```json
{
  "backbone": "CBraMod",
  "backbone_frozen": true,
  "dataset": "BCIC-IV-2a",
  "decision": "SOFTWEIGHT_NEXT",
  "interpretation_level": "DIAGNOSTIC_ONLY",
  "layer0_smoke": {
    "cbramod_input_shape": [
      22,
      4,
      200
    ],
    "feature_shape": [
      200
    ],
    "passed": true,
    "raw_shape": [
      22,
      800
    ]
  },
  "layer1": {
    "bad_artifact_delta_acc_vs_clean": -0.0054012345679012586,
    "bad_content_delta_acc_vs_clean": -0.06378600823045272,
    "bad_physio_delta_acc_vs_clean": -0.0007716049382716084,
    "bad_style_delta_acc_vs_clean": -0.009516460905349855,
    "passed": false
  },
  "layer2": {
    "bad_artifact_auc": 0.947394718792867,
    "bad_content_auc": 0.8008573388203017,
    "bad_physio_auc": 0.15951303155006857,
    "bad_style_auc": 0.5789948559670782,
    "bottom30_bad_rate": 0.5747599451303155,
    "overall_auc": 0.6216899862825789,
    "passed": false,
    "top30_bad_rate": 0.3628257887517147
  },
  "layer3": {
    "passed": false,
    "sas_top50_minus_naive_acc": -0.02366255144032922,
    "sas_top50_minus_naive_ece": -0.013917024900807153,
    "sas_top50_minus_random50_acc": 0.00913065843621399,
    "subject_std_change": null,
    "worst_subject_gain": null
  },
  "next_action": "Refine artifact/content/physio certificate and run SoftWeight/ArtifactReject shadow groups.",
  "project": "sas_cert_cbramod_mve",
  "protocol": "FEWSHOT_TARGET_SUPPORT",
  "protocol_leakage_detected": false,
  "seeds": [
    20,
    21,
    22
  ],
  "shot_per_class": 5,
  "status": "completed",
  "subjects": 9
}
```
