#!/usr/bin/env python3
"""Audit direct-use EEG reference algorithms for SAS-Cert v1.1.

This runner intentionally avoids model training and raw-data copying. It clones
only whitelisted repositories, uses an isolated virtual environment for install
checks, runs tiny synthetic-data smoke tests, and writes audit artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import venv
from pathlib import Path


ROOT = Path("/ai/224duibishiyan/615新研究")
THIRD_PARTY = ROOT / "third_party" / "reference_algorithms"
OUT_DIR = ROOT / "outputs" / "reference_algorithm_audit"
VENV_DIR = OUT_DIR / "audit_venv"
PY_TARGET = OUT_DIR / "python_target"


REPOS = [
    {
        "project_name": "MNE-ICALabel",
        "repo_url": "https://github.com/mne-tools/mne-icalabel",
        "dirname": "mne-icalabel",
        "sparse": None,
    },
    {
        "project_name": "Autoreject",
        "repo_url": "https://github.com/autoreject/autoreject",
        "dirname": "autoreject",
        "sparse": None,
    },
    {
        "project_name": "pyRiemann",
        "repo_url": "https://github.com/pyRiemann/pyRiemann",
        "dirname": "pyRiemann",
        "sparse": None,
    },
    {
        "project_name": "MNE-Features",
        "repo_url": "https://github.com/mne-tools/mne-features",
        "dirname": "mne-features",
        "sparse": None,
    },
    {
        "project_name": "Braindecode",
        "repo_url": "https://github.com/braindecode/braindecode",
        "dirname": "braindecode",
        "sparse": None,
    },
    {
        "project_name": "EEG-DLite",
        "repo_url": "https://github.com/t170815518/EEG-DLite",
        "dirname": "EEG-DLite",
        "sparse": None,
    },
    {
        "project_name": "MOABB",
        "repo_url": "https://github.com/NeuroTechX/moabb",
        "dirname": "moabb",
        "sparse": None,
    },
    {
        "project_name": "Channel Reflection",
        "repo_url": "https://github.com/wzwvv/EEGAug",
        "dirname": "EEGAug_ChannelReflection_sparse",
        "sparse": [
            "README.md",
            "LICENSE",
            "requirements.txt",
            "environment.yml",
            "within_CR.py",
            "within_baseline.py",
            "utils",
        ],
    },
]


PIP_PACKAGES = {
    "pyRiemann": ["pyriemann"],
    "MNE-Features": ["mne-features", "PyWavelets"],
    "Autoreject": ["autoreject", "h5io"],
    "MNE-ICALabel": ["mne-icalabel"],
    "Braindecode": ["braindecode", "skorch"],
    "MOABB": ["moabb"],
}


SMOKE_SNIPPETS = {
    "pyRiemann": r"""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
rng = np.random.RandomState(0)
X = rng.randn(5, 4, 128)
covs = Covariances(estimator="scm").fit_transform(X)
d = float(distance_riemann(covs[0], covs[1]))
assert covs.shape == (5, 4, 4)
assert np.isfinite(d)
print({"cov_shape": covs.shape, "distance_riemann": d})
""",
    "MNE-Features": r"""
import numpy as np
import mne_features
from mne_features.feature_extraction import extract_features
rng = np.random.RandomState(0)
X = rng.randn(3, 4, 128)
sfreq = 128.0
selected = ["mean", "std", "kurtosis", "skewness"]
features = extract_features(X, sfreq, selected, n_jobs=1, return_as_df=False)
assert features.shape[0] == 3
print({"feature_shape": features.shape, "selected": selected})
""",
    "Autoreject": r"""
import numpy as np
import mne
from autoreject import AutoReject
rng = np.random.RandomState(0)
ch_names = ["Fp1", "Fp2", "C3", "C4"]
info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
info.set_montage("standard_1020")
data = rng.randn(12, len(ch_names), 128) * 1e-6
events = np.column_stack([np.arange(12), np.zeros(12, dtype=int), np.ones(12, dtype=int)])
epochs = mne.EpochsArray(data, info, events=events, event_id={"tiny": 1}, tmin=0.0, verbose=False)
ar = AutoReject(n_interpolate=[1], consensus=[0.5, 0.8], cv=2, random_state=0, verbose=False)
ar.fit(epochs)
reject_log = ar.get_reject_log(epochs)
print({"bad_epoch_count": int(reject_log.bad_epochs.sum()), "bad_epoch_shape": reject_log.bad_epochs.shape})
""",
    "MNE-ICALabel": r"""
import numpy as np
import mne
import mne_icalabel
from mne.preprocessing import ICA
from mne_icalabel import label_components
rng = np.random.RandomState(0)
ch_names = ["Fp1", "Fp2", "Fz", "C3", "C4", "Cz", "Pz", "Oz"]
info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
info.set_montage("standard_1020")
raw = mne.io.RawArray(rng.randn(len(ch_names), 2048) * 1e-6, info, verbose=False)
raw.set_eeg_reference("average", projection=False, verbose=False)
ica = ICA(n_components=4, random_state=0, max_iter=80, method="fastica")
ica.fit(raw, verbose=False)
labels = label_components(raw, ica, method="iclabel")
assert "labels" in labels
print({"labels": labels.get("labels"), "keys": sorted(labels.keys())})
""",
    "Braindecode": r"""
import torch
import braindecode
import braindecode.augmentation as aug
names = [n for n in dir(aug) if any(k in n.lower() for k in ["noise", "dropout", "mask", "frequency", "reverse", "flip"])]
result = {"augmentation_names": names[:25]}
X = torch.randn(3, 4, 128)
y = torch.tensor([0, 1, 0])
if hasattr(aug, "GaussianNoise"):
    transform = aug.GaussianNoise(probability=1.0, std=0.01)
    out = transform(X, y)
    result["gaussian_noise_type"] = str(type(out))
print(result)
""",
    "MOABB": r"""
import moabb
from moabb import datasets
names = [n for n in dir(datasets) if "Physionet" in n or "Motor" in n or "MI" in n]
print({"moabb_version": getattr(moabb, "__version__", "unknown"), "candidate_datasets": names[:30]})
""",
}


def run(cmd, cwd=None, timeout=300, env=None):
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 2),
            "output": proc.stdout[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "elapsed_sec": round(time.time() - started, 2),
            "output": (exc.stdout or "")[-4000:] + "\nTIMEOUT",
        }


def ensure_dirs():
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def clone_repo(repo):
    path = THIRD_PARTY / repo["dirname"]
    if path.exists() and (path / ".git").exists():
        status = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], timeout=60)
        return path, "already_present:" + status["output"].strip()
    if path.exists():
        return path, "blocked:local_path_exists_not_git"
    if repo["sparse"]:
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            repo["repo_url"],
            str(path),
        ]
        res = run(cmd, timeout=600)
        if res["returncode"] != 0:
            return path, "clone_failed:" + res["output"]
        init = run(["git", "-C", str(path), "sparse-checkout", "init", "--no-cone"], timeout=120)
        if init["returncode"] != 0:
            return path, "sparse_init_failed:" + init["output"]
        sparse_patterns = []
        for item in repo["sparse"]:
            sparse_patterns.append(item)
            if not item.endswith("*") and "." not in Path(item).name:
                sparse_patterns.append(item.rstrip("/") + "/*")
        sparse = run(["git", "-C", str(path), "sparse-checkout", "set"] + sparse_patterns, timeout=120)
        if sparse["returncode"] != 0:
            return path, "sparse_failed:" + sparse["output"]
        checkout = run(["git", "-C", str(path), "checkout"], timeout=120)
        if checkout["returncode"] != 0:
            return path, "sparse_checkout_failed:" + checkout["output"]
        return path, "sparse_cloned_no_data_dir"
    res = run(["git", "clone", "--depth", "1", repo["repo_url"], str(path)], timeout=900)
    if res["returncode"] == 0:
        return path, "cloned"
    return path, "clone_failed:" + res["output"]


def create_python_target():
    if PY_TARGET.exists():
        shutil.rmtree(PY_TARGET)
    PY_TARGET.mkdir(parents=True, exist_ok=True)
    py = Path(sys.executable)
    pip = [str(py), "-m", "pip"]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PY_TARGET) + (os.pathsep + existing if existing else "")
    return py, pip, env


def pip_install(project_name, pip_cmd):
    packages = PIP_PACKAGES.get(project_name)
    if not packages:
        return "not_applicable", ""
    res = run(
        pip_cmd
        + [
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--no-deps",
            "--target",
            str(PY_TARGET),
        ]
        + packages,
        timeout=240,
    )
    if res["returncode"] == 0:
        return "pip_install_no_deps_ok:" + " ".join(packages), res["output"]
    return "pip_install_no_deps_failed:" + " ".join(packages), res["output"]


def smoke(project_name, py, env):
    snippet = SMOKE_SNIPPETS.get(project_name)
    if not snippet:
        return "not_applicable", ""
    res = run([str(py), "-c", snippet], timeout=300, env=env)
    if res["returncode"] == 0:
        return "smoke_ok", res["output"]
    return "smoke_failed", res["output"]


def import_check(project_name, py, env):
    modules = {
        "MNE-ICALabel": ["mne_icalabel"],
        "Autoreject": ["autoreject"],
        "pyRiemann": ["pyriemann"],
        "MNE-Features": ["mne_features"],
        "Braindecode": ["braindecode"],
        "MOABB": ["moabb"],
    }.get(project_name, [])
    if not modules:
        return "not_applicable", ""
    code = "\n".join(
        [
            "import importlib, json",
            f"mods={modules!r}",
            "out={}",
            "for m in mods:",
            "    mod=importlib.import_module(m)",
            "    out[m]=getattr(mod, '__version__', 'unknown')",
            "print(json.dumps(out, sort_keys=True))",
        ]
    )
    res = run([str(py), "-c", code], timeout=120, env=env)
    if res["returncode"] == 0:
        return "import_ok", res["output"].strip()
    return "import_failed", res["output"]


def inspect_repo(project_name, path):
    notes = []
    required_input = ""
    output_format = ""
    dependency_risk = "medium"
    runtime_risk = "medium"
    if not path.exists():
        return required_input, output_format, dependency_risk, runtime_risk, "repo_missing"

    files = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
    top = set(files)
    if "requirements.txt" in top:
        notes.append("has_requirements")
    if "pyproject.toml" in top:
        notes.append("has_pyproject")
    if "setup.py" in top:
        notes.append("has_setup_py")

    if project_name == "MNE-ICALabel":
        api_hits = [f for f in files if "label_components" in f or "iclabel" in f.lower()]
        required_input = "MNE Raw/Epochs plus fitted MNE ICA"
        output_format = "component labels and class probabilities when smoke/API succeeds"
        dependency_risk = "medium"
        runtime_risk = "high: requires ICA fitting and suitable preprocessing"
        notes.append("api_files_hint=" + ",".join(api_hits[:5]))
    elif project_name == "Autoreject":
        required_input = "MNE Epochs with channel positions/montage"
        output_format = "reject_log, bad_epochs, repair/interpolation information"
        dependency_risk = "medium"
        runtime_risk = "medium: cross-validation over epochs"
    elif project_name == "pyRiemann":
        required_input = "trial tensor [N,C,T] or covariance matrices [N,C,C]"
        output_format = "SPD covariance matrices and Riemannian distances"
        dependency_risk = "low"
        runtime_risk = "low"
    elif project_name == "MNE-Features":
        required_input = "trial tensor [N,C,T] with sampling frequency"
        output_format = "tabular spectral/statistical features per trial"
        dependency_risk = "low-medium"
        runtime_risk = "low"
    elif project_name == "Braindecode":
        required_input = "torch/numpy EEG trials, usually [N,C,T]"
        output_format = "augmentation transforms and baseline model utilities"
        dependency_risk = "high: torch/skorch/moabb ecosystem"
        runtime_risk = "low for transforms, high for training utilities"
    elif project_name == "EEG-DLite":
        has_distill = any(f.endswith("distillate_datasets.py") for f in files)
        hdf5_hits = [f for f in files if "hdf" in f.lower() or "h5" in f.lower()]
        required_input = "likely pretraining dataset files/features; inspect before use"
        output_format = "distilled/filtered dataset subsets"
        dependency_risk = "medium-high"
        runtime_risk = "medium-high"
        notes.append("has_distillate_datasets_py=" + str(has_distill))
        notes.append("hdf5_related_files=" + ",".join(hdf5_hits[:8]))
    elif project_name == "MOABB":
        required_input = "MOABB dataset loaders/pipelines"
        output_format = "benchmark datasets and pipeline results"
        dependency_risk = "high: benchmark stack and optional dataset downloads"
        runtime_risk = "high if datasets are downloaded/evaluated"
    elif project_name == "Channel Reflection":
        has_data = any(f.startswith("data/") for f in files)
        required_input = "EEG trials with known left/right channel mapping"
        output_format = "reflected EEG trials, label-swapped for MI when appropriate"
        dependency_risk = "medium"
        runtime_risk = "low for transform, high if reproducing paper scripts"
        notes.append("sparse_clone_excludes_data_dir=" + str(not has_data))
        notes.append("official_repo_readme_claims_CR_implementation")

    return required_input, output_format, dependency_risk, runtime_risk, "; ".join(notes)


def direct_use_level(row):
    name = row["project_name"]
    smoke_status = row["smoke_status"]
    import_status = row["import_status"]
    install_status = row["install_status"]
    if "failed" in install_status or "failed" in import_status:
        if name in {"EEG-DLite", "MOABB", "Channel Reflection"}:
            return "cite_only"
        return "blocked"
    if name in {"pyRiemann", "MNE-Features"} and smoke_status == "smoke_ok":
        return "use_now"
    if name in {"Autoreject", "MNE-ICALabel"}:
        if import_status == "import_ok" and smoke_status == "smoke_ok":
            return "use_offline_only"
        if import_status == "import_ok":
            return "use_offline_only"
        return "blocked"
    if name == "Braindecode":
        if import_status == "import_ok" and smoke_status == "smoke_ok":
            return "use_later"
        return "blocked"
    if name == "EEG-DLite":
        return "use_later"
    if name == "MOABB":
        return "use_later" if import_status == "import_ok" else "cite_only"
    if name == "Channel Reflection":
        return "use_later"
    return "cite_only"


def useful_for(name):
    return {
        "MNE-ICALabel": "artifact_probability/safety_gate",
        "Autoreject": "safety_gate",
        "pyRiemann": "physio_score/style_score",
        "MNE-Features": "safety_gate/physio_score/style_score",
        "Braindecode": "augmentation_pool/baseline",
        "EEG-DLite": "augmentation_pool/redundancy_filtering",
        "MOABB": "benchmark",
        "Channel Reflection": "augmentation_pool",
    }[name]


def write_outputs(rows, logs):
    json_rows = OUT_DIR / "reference_algorithm_inventory.json"
    csv_rows = OUT_DIR / "reference_algorithm_inventory.csv"
    compact = OUT_DIR / "compact_reference_algorithm_result.json"
    report = OUT_DIR / "REFERENCE_ALGORITHM_AUDIT_REPORT.md"
    blueprint = OUT_DIR / "sascert_softar_ls_v1_1_blueprint.md"
    logs_path = OUT_DIR / "audit_command_logs.json"

    fieldnames = [
        "project_name",
        "repo_url",
        "local_path",
        "install_status",
        "import_status",
        "smoke_status",
        "direct_use_level",
        "useful_for",
        "required_input_format",
        "output_format",
        "dependency_risk",
        "runtime_risk",
        "notes",
    ]

    with csv_rows.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_rows.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logs_path.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")

    by_level = {}
    for row in rows:
        by_level.setdefault(row["direct_use_level"], []).append(row["project_name"])

    compact_obj = {
        "task": "SASCERT_DIRECT_USE_ALGORITHM_AUDIT_AND_V1_BLUEPRINT",
        "completed": True,
        "output_dir": str(OUT_DIR),
        "third_party_dir": str(THIRD_PARTY),
        "use_now": by_level.get("use_now", []),
        "use_offline_only": by_level.get("use_offline_only", []),
        "use_later": by_level.get("use_later", []),
        "cite_only": by_level.get("cite_only", []),
        "blocked": by_level.get("blocked", []),
        "no_training": True,
        "no_raw_data_copy": True,
        "channel_reflection_source_note": (
            "wzwvv/EEGAug README claims official implementation; arXiv HTML v1 "
            "also footnotes sylyoung/DeepTransferEEG. Sparse clone used for "
            "wzwvv/EEGAug and data/ was excluded."
        ),
        "next_experiment_allowed_after_user_approval": (
            "ST-EEGFormer-small + PhysioNetMI: NaiveAug_LS010, "
            "ArtifactReject_LS010, SoftWeight_noReject_LS010, "
            "SAS-Cert-SoftAR-LS-v1.1"
        ),
    }
    compact.write_text(json.dumps(compact_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def bullet(names):
        return "\n".join(f"- {n}" for n in names) if names else "- None"

    report.write_text(
        textwrap.dedent(
            f"""\
            # Reference Algorithm Audit Report

            Task: `SASCERT_DIRECT_USE_ALGORITHM_AUDIT_AND_V1_BLUEPRINT`

            ## Scope

            This audit downloaded and inspected only whitelisted reference
            projects for SAS-Cert-SoftAR-LS v1.1. It did not train any model,
            run a SAS-Cert experiment, copy raw EEG data, or modify old
            experimental outputs. Installation checks used an isolated
            `pip --target` directory:

            `{PY_TARGET}`

            Third-party reference code was placed under:

            `{THIRD_PARTY}`

            Reports were written under:

            `{OUT_DIR}`

            ## Summary By Recommendation

            ### use_now

            {bullet(by_level.get("use_now", []))}

            ### use_offline_only

            {bullet(by_level.get("use_offline_only", []))}

            ### use_later

            {bullet(by_level.get("use_later", []))}

            ### cite_only

            {bullet(by_level.get("cite_only", []))}

            ### blocked

            {bullet(by_level.get("blocked", []))}

            ## Inventory

            | Project | Install | Import | Smoke | Recommendation | Useful for |
            |---|---|---|---|---|---|
            """
        )
        + "\n".join(
            "| {project_name} | {install_status} | {import_status} | {smoke_status} | {direct_use_level} | {useful_for} |".format(
                **row
            )
            for row in rows
        )
        + textwrap.dedent(
            """

            ## Channel Reflection Source Note

            Search found `https://github.com/wzwvv/EEGAug`, whose README states
            it is the official implementation of Channel Reflection. The arXiv
            HTML v1 page also footnotes `https://github.com/sylyoung/DeepTransferEEG`.
            This audit used the README-identified `wzwvv/EEGAug` repository and
            performed a sparse clone excluding `data/` to avoid copying EEG data.

            ## Interpretation

            `pyRiemann` and `MNE-Features`, when smoke checks pass, are the
            strongest direct v1.1 dependencies because they operate on synthetic
            trial tensors or covariance/features without requiring dataset
            loaders. `Autoreject` and `MNE-ICALabel` should be treated as
            offline safety/audit tools first because they require MNE Epochs/Raw,
            channel metadata, and in the ICLabel case a fitted ICA. `Braindecode`
            is useful for augmentation and baselines but should not determine the
            SAS-Cert logic. `EEG-DLite`, `MOABB`, and Channel Reflection are
            useful references or later candidates, not required v1.1 components.
            """
        ),
        encoding="utf-8",
    )

    blueprint.write_text(
        textwrap.dedent(
            """\
            # SAS-Cert-SoftAR-LS v1.1 Blueprint

            ## Unified Algorithm Logic

            SAS-Cert-SoftAR-LS v1.1 should remain a risk-controlled
            augmentation-utilization policy:

            1. Safety Gate
            2. Label-Preservation Evidence
            3. Physiology/Style Plausibility
            4. Utility Weight
            5. Calibration-aware Training

            The algorithm is not a flat sum of external modules. Each reference
            tool supports one decision layer, while SAS-Cert keeps the decision
            order and training policy.

            ## Layer 1: Safety Gate

            Purpose: reject or downweight augmented trials that are likely
            unsafe because of artifact or severe signal-quality failure.

            Minimal v1.1:

            - Primary implementation: existing rule artifact score using
              high-frequency energy, low-frequency drift, line-noise power,
              channel energy outliers, kurtosis, and skewness.
            - Optional offline audit: Autoreject reject logs on MNE Epochs.
            - Optional offline audit: MNE-ICALabel component probabilities after
              fitted ICA.
            - Optional feature support: MNE-Features for trial-level statistics.

            Policy:

            - Highest artifact-risk decile is rejected: `w = 0`.
            - Artifact risk is a qualification gate, not a positive additive
              score.

            ## Layer 2: Label-Preservation Evidence

            Purpose: test whether `x_aug` still preserves the label semantics of
            the original trial.

            Inputs:

            - ST-EEGFormer-small embedding.
            - CBraMod embedding.
            - Current classifier prediction on `x` and `x_aug`.

            Evidence:

            ```text
            E_embed = cosine(f(x), f(x_aug))
            E_proto = cosine(f(x_aug), prototype_y)
                    - max_{c != y} cosine(f(x_aug), prototype_c)
            E_pred = -KL(p(.|x) || p(.|x_aug))
            E_content = ranknorm(E_embed) + ranknorm(E_proto)
                      + 0.5 * ranknorm(E_pred)
            ```

            SCOPE/EEGTune are design references for agreement and prediction
            stability, but their pseudo-label systems are not imported into
            v1.1.

            ## Layer 3: Physiology/Style Plausibility

            Purpose: ensure that label-like samples do not violate EEG
            physiology or target-subject style.

            Direct tools:

            - pyRiemann for covariance matrices and Riemannian distances.
            - MNE-Features for bandpower/statistical features.

            Minimal v1.1:

            ```text
            D_band = bandpower_deviation(x_aug, x_or_source_reference)
            D_cov = riemannian_covariance_distance(x_aug, reference)
            E_physio = 1 - ranknorm(D_band + D_cov)

            style_target = target support mean/std + bandpower + covariance summary
            style_aug = same summary for x_aug
            D_style = distance(style_aug, style_target)
            E_style = 1 - ranknorm(D_style)
            ```

            Style remains auxiliary because previous audits showed that style can
            be unstable across backbones and bad types.

            ## Layer 4: Utility Weight

            For samples that pass the safety gate:

            ```text
            score = E_content + E_physio + 0.5 * E_style
            w = 0.2 + 0.8 * ranknorm(score)
            ```

            For rejected samples:

            ```text
            w = 0
            ```

            ## Layer 5: Calibration-aware Training

            Minimal loss:

            ```text
            L = CE(real_support, y; label_smoothing=0.10)
              + mean_i w_i * CE(x_aug_i, y_i; label_smoothing=0.10)
            ```

            Label smoothing is fixed at `0.10` for the v1.1 comparison, matching
            the current LS010 branch.

            ## Tools Not Entering v1.1 As Required Dependencies

            - EEG-DLite: useful for later outlier/redundancy filtering, but its
              native workflow is data-distillation/pretraining-oriented and
              should not block v1.1.
            - MOABB: useful for benchmark protocol and loaders, but not part of
              the core algorithm.
            - Channel Reflection: useful knowledge-driven augmentation candidate
              for a later augmentation-pool expansion; not required for the
              initial v1.1 reliability policy.
            - Braindecode: useful augmentation/baseline utilities, but should
              not define SAS-Cert's decision logic.
            - HAPPE/PREP/ADJUST/ArtifactGen: cite or later only; do not force
              MATLAB/EEGLAB or heavy generative systems into v1.1.

            ## Next Experiment After User Approval

            Dataset/backbone:

            - ST-EEGFormer-small + PhysioNetMI left-vs-right MI, runs R04/R08/R12.

            Four groups:

            - NaiveAug_LS010
            - ArtifactReject_LS010
            - SoftWeight_noReject_LS010
            - SAS-Cert-SoftAR-LS-v1.1

            Promotion gate:

            - v1.1 must beat or match the relevant baselines on balanced
              accuracy/Macro-F1 without worsening ECE/NLL/Brier, and it must
              improve subject/seed reliability rather than only mean metrics.
            """
        ),
        encoding="utf-8",
    )


def main():
    ensure_dirs()
    rows = []
    logs = {"clone": {}, "install": {}, "import": {}, "smoke": {}}

    py, pip, env = create_python_target()
    logs["python"] = str(py)
    logs["python_target"] = str(PY_TARGET)

    for repo in REPOS:
        name = repo["project_name"]
        local_path, clone_status = clone_repo(repo)
        logs["clone"][name] = clone_status

        install_status, install_log = pip_install(name, pip)
        logs["install"][name] = install_log[-8000:]

        import_status, import_log = import_check(name, py, env)
        logs["import"][name] = import_log[-8000:]

        smoke_status, smoke_log = smoke(name, py, env)
        logs["smoke"][name] = smoke_log[-8000:]

        req_in, out_fmt, dep_risk, run_risk, notes = inspect_repo(name, local_path)
        row = {
            "project_name": name,
            "repo_url": repo["repo_url"],
            "local_path": str(local_path),
            "install_status": install_status,
            "import_status": import_status,
            "smoke_status": smoke_status,
            "direct_use_level": "",
            "useful_for": useful_for(name),
            "required_input_format": req_in,
            "output_format": out_fmt,
            "dependency_risk": dep_risk,
            "runtime_risk": run_risk,
            "notes": f"clone_status={clone_status}; {notes}".strip(),
        }
        row["direct_use_level"] = direct_use_level(row)
        rows.append(row)

    write_outputs(rows, logs)
    print(json.dumps({"completed": True, "output_dir": str(OUT_DIR), "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
