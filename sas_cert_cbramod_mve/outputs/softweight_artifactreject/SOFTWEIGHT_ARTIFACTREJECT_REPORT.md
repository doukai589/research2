# SoftWeight / ArtifactReject Shadow Validation

```json
{
  "decision": {
    "best_group_by_acc_vs_naive": "ArtifactReject",
    "best_passing_shadow_group": "SoftWeightArtifactReject",
    "group_decisions": {
      "ArtifactReject": {
        "acc_vs_naive": 0.016846707818930048,
        "acc_vs_top50": 0.01916152263374485,
        "ece_vs_naive": 0.017037372025297072,
        "macro_f1_vs_naive": 0.03318337618842063,
        "passes_shadow_gate": false
      },
      "SASCertSoftWeight": {
        "acc_vs_naive": 0.007844650205761326,
        "acc_vs_top50": 0.010159465020576127,
        "ece_vs_naive": 0.007409120141432731,
        "macro_f1_vs_naive": 0.0012821990623061153,
        "passes_shadow_gate": true
      },
      "SoftWeightArtifactReject": {
        "acc_vs_naive": 0.011574074074074079,
        "acc_vs_top50": 0.01388888888888888,
        "ece_vs_naive": 0.008254898930497375,
        "macro_f1_vs_naive": 0.014293837205182446,
        "passes_shadow_gate": true
      }
    },
    "recommended_next": "PROMOTE_SOFTWEIGHT_SHADOW_TO_MAIN"
  },
  "groups": [
    "ArtifactReject",
    "NaiveAug",
    "NoAug",
    "Random50",
    "SASCertSoftWeight",
    "SASCertTop50",
    "SoftWeightArtifactReject"
  ],
  "paired_comparison": [
    {
      "comparison": "SASCertSoftWeight_minus_NaiveAug",
      "mean_delta": 0.007844650205761326,
      "metric": "acc",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "SASCertSoftWeight_minus_NaiveAug",
      "mean_delta": 0.0012821990623061153,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "SASCertSoftWeight_minus_NaiveAug",
      "mean_delta": 0.007409120141432731,
      "metric": "ece",
      "n": 27,
      "positive_folds": 17
    },
    {
      "comparison": "SASCertSoftWeight_minus_NaiveAug",
      "mean_delta": -0.00023843182457817925,
      "metric": "nll",
      "n": 27,
      "positive_folds": 12
    },
    {
      "comparison": "SASCertSoftWeight_minus_Random50",
      "mean_delta": 0.027777777777777776,
      "metric": "acc",
      "n": 27,
      "positive_folds": 21
    },
    {
      "comparison": "SASCertSoftWeight_minus_Random50",
      "mean_delta": 0.039971437239628205,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 20
    },
    {
      "comparison": "SASCertSoftWeight_minus_Random50",
      "mean_delta": -0.0040179132253943435,
      "metric": "ece",
      "n": 27,
      "positive_folds": 10
    },
    {
      "comparison": "SASCertSoftWeight_minus_Random50",
      "mean_delta": -0.008294026056925455,
      "metric": "nll",
      "n": 27,
      "positive_folds": 7
    },
    {
      "comparison": "SASCertSoftWeight_minus_NoAug",
      "mean_delta": -0.002829218106995885,
      "metric": "acc",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "SASCertSoftWeight_minus_NoAug",
      "mean_delta": -0.014501338312320873,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SASCertSoftWeight_minus_NoAug",
      "mean_delta": -0.005043975597845367,
      "metric": "ece",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SASCertSoftWeight_minus_NoAug",
      "mean_delta": -0.0006563707634254739,
      "metric": "nll",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "SASCertSoftWeight_minus_SASCertTop50",
      "mean_delta": 0.010159465020576127,
      "metric": "acc",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "SASCertSoftWeight_minus_SASCertTop50",
      "mean_delta": 0.015161426236225937,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 18
    },
    {
      "comparison": "SASCertSoftWeight_minus_SASCertTop50",
      "mean_delta": 0.0070962999840056195,
      "metric": "ece",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "SASCertSoftWeight_minus_SASCertTop50",
      "mean_delta": -0.00340661296138057,
      "metric": "nll",
      "n": 27,
      "positive_folds": 5
    },
    {
      "comparison": "ArtifactReject_minus_NaiveAug",
      "mean_delta": 0.016846707818930048,
      "metric": "acc",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "ArtifactReject_minus_NaiveAug",
      "mean_delta": 0.03318337618842063,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "ArtifactReject_minus_NaiveAug",
      "mean_delta": 0.017037372025297072,
      "metric": "ece",
      "n": 27,
      "positive_folds": 19
    },
    {
      "comparison": "ArtifactReject_minus_NaiveAug",
      "mean_delta": -0.0012617817631474248,
      "metric": "nll",
      "n": 27,
      "positive_folds": 8
    },
    {
      "comparison": "ArtifactReject_minus_Random50",
      "mean_delta": 0.0367798353909465,
      "metric": "acc",
      "n": 27,
      "positive_folds": 19
    },
    {
      "comparison": "ArtifactReject_minus_Random50",
      "mean_delta": 0.07187261436574273,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 23
    },
    {
      "comparison": "ArtifactReject_minus_Random50",
      "mean_delta": 0.005610338658469999,
      "metric": "ece",
      "n": 27,
      "positive_folds": 15
    },
    {
      "comparison": "ArtifactReject_minus_Random50",
      "mean_delta": -0.009317375995494702,
      "metric": "nll",
      "n": 27,
      "positive_folds": 5
    },
    {
      "comparison": "ArtifactReject_minus_NoAug",
      "mean_delta": 0.006172839506172838,
      "metric": "acc",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "ArtifactReject_minus_NoAug",
      "mean_delta": 0.017399838813793648,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "ArtifactReject_minus_NoAug",
      "mean_delta": 0.004584276286018977,
      "metric": "ece",
      "n": 27,
      "positive_folds": 12
    },
    {
      "comparison": "ArtifactReject_minus_NoAug",
      "mean_delta": -0.0016797207019947193,
      "metric": "nll",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "ArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.01916152263374485,
      "metric": "acc",
      "n": 27,
      "positive_folds": 16
    },
    {
      "comparison": "ArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.04706260336234044,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 19
    },
    {
      "comparison": "ArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.01672455186786996,
      "metric": "ece",
      "n": 27,
      "positive_folds": 19
    },
    {
      "comparison": "ArtifactReject_minus_SASCertTop50",
      "mean_delta": -0.004429962899949815,
      "metric": "nll",
      "n": 27,
      "positive_folds": 6
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NaiveAug",
      "mean_delta": 0.011574074074074079,
      "metric": "acc",
      "n": 27,
      "positive_folds": 17
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NaiveAug",
      "mean_delta": 0.014293837205182446,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 18
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NaiveAug",
      "mean_delta": 0.008254898930497375,
      "metric": "ece",
      "n": 27,
      "positive_folds": 18
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NaiveAug",
      "mean_delta": -0.0008749343730785229,
      "metric": "nll",
      "n": 27,
      "positive_folds": 11
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_Random50",
      "mean_delta": 0.031507201646090534,
      "metric": "acc",
      "n": 27,
      "positive_folds": 20
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_Random50",
      "mean_delta": 0.052983075382504546,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 21
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_Random50",
      "mean_delta": -0.0031721344363296986,
      "metric": "ece",
      "n": 27,
      "positive_folds": 14
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_Random50",
      "mean_delta": -0.0089305286054258,
      "metric": "nll",
      "n": 27,
      "positive_folds": 6
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NoAug",
      "mean_delta": 0.0009002057613168671,
      "metric": "acc",
      "n": 27,
      "positive_folds": 11
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NoAug",
      "mean_delta": -0.0014897001694445379,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NoAug",
      "mean_delta": -0.004198196808780721,
      "metric": "ece",
      "n": 27,
      "positive_folds": 11
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_NoAug",
      "mean_delta": -0.0012928733119258174,
      "metric": "nll",
      "n": 27,
      "positive_folds": 12
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.01388888888888888,
      "metric": "acc",
      "n": 27,
      "positive_folds": 15
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.028173064379102268,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 21
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_SASCertTop50",
      "mean_delta": 0.007942078773070266,
      "metric": "ece",
      "n": 27,
      "positive_folds": 20
    },
    {
      "comparison": "SoftWeightArtifactReject_minus_SASCertTop50",
      "mean_delta": -0.0040431155098809134,
      "metric": "nll",
      "n": 27,
      "positive_folds": 5
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": -0.0023148148148148004,
      "metric": "acc",
      "n": 27,
      "positive_folds": 13
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": -0.013879227173919819,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 11
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": 0.0003128201574271101,
      "metric": "ece",
      "n": 27,
      "positive_folds": 15
    },
    {
      "comparison": "SASCertTop50_minus_NaiveAug",
      "mean_delta": 0.003168181136802391,
      "metric": "nll",
      "n": 27,
      "positive_folds": 18
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": 0.017618312757201653,
      "metric": "acc",
      "n": 27,
      "positive_folds": 21
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": 0.024810011003402275,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 19
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": -0.011114213209399962,
      "metric": "ece",
      "n": 27,
      "positive_folds": 12
    },
    {
      "comparison": "SASCertTop50_minus_Random50",
      "mean_delta": -0.0048874130955448855,
      "metric": "nll",
      "n": 27,
      "positive_folds": 8
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.012988683127572011,
      "metric": "acc",
      "n": 27,
      "positive_folds": 11
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.029662764548546803,
      "metric": "macro_f1",
      "n": 27,
      "positive_folds": 12
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": -0.012140275581850988,
      "metric": "ece",
      "n": 27,
      "positive_folds": 9
    },
    {
      "comparison": "SASCertTop50_minus_NoAug",
      "mean_delta": 0.0027502421979550963,
      "metric": "nll",
      "n": 27,
      "positive_folds": 18
    }
  ],
  "phase": "softweight_artifactreject_shadow_validation",
  "protocol_leakage_detected": false,
  "source": "cached_original_and_augmented_cbramod_features_from_main_mve",
  "status": "completed",
  "summary": [
    {
      "acc": 0.3265174897119342,
      "ece": 0.07443784687231168,
      "group": "ArtifactReject",
      "kappa": 0.1020233196159122,
      "macro_f1": 0.26748662667281004,
      "n": 27,
      "nll": 1.3683700340765494
    },
    {
      "acc": 0.3096707818930041,
      "ece": 0.05740047484701461,
      "group": "NaiveAug",
      "kappa": 0.07956104252400548,
      "macro_f1": 0.23430325048438938,
      "n": 27,
      "nll": 1.3696318158396967
    },
    {
      "acc": 0.3203446502057613,
      "ece": 0.06985357058629271,
      "group": "NoAug",
      "kappa": 0.09379286694101509,
      "macro_f1": 0.25008678785901634,
      "n": 27,
      "nll": 1.3700497547785442
    },
    {
      "acc": 0.2897376543209877,
      "ece": 0.06882750821384169,
      "group": "Random50",
      "kappa": 0.05298353909465021,
      "macro_f1": 0.1956140123070673,
      "n": 27,
      "nll": 1.377687410072044
    },
    {
      "acc": 0.3175154320987654,
      "ece": 0.06480959498844735,
      "group": "SASCertSoftWeight",
      "kappa": 0.09002057613168725,
      "macro_f1": 0.23558544954669552,
      "n": 27,
      "nll": 1.3693933840151187
    },
    {
      "acc": 0.3073559670781893,
      "ece": 0.057713295004441716,
      "group": "SASCertTop50",
      "kappa": 0.07647462277091907,
      "macro_f1": 0.22042402331046956,
      "n": 27,
      "nll": 1.3727999969764992
    },
    {
      "acc": 0.32124485596707825,
      "ece": 0.06565537377751197,
      "group": "SoftWeightArtifactReject",
      "kappa": 0.09499314128943759,
      "macro_f1": 0.24859708768957187,
      "n": 27,
      "nll": 1.3687568814666182
    }
  ]
}
```
