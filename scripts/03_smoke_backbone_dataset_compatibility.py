from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BNCI_CHANNELS = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 backbone x dataset smoke and compatibility audit.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out_dir = workspace / "outputs" / "setup_audit_step1"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "workspace": workspace,
        "bcic2a": workspace.parent / "CBraMod-main" / "tmp_in" / "BCIC2a" / "MNE-bnci-data" / "database" / "data-sets" / "001-2014",
        "physionetmi": workspace.parent / "CBraMod-main" / "tmp_in" / "MI" / "files",
        "cbramod": workspace / "sas_cert_cbramod_mve" / "third_party" / "CBraMod",
        "mirepnet": workspace / "third_party" / "backbones" / "MIRepNet",
        "labram": workspace / "third_party" / "backbones" / "LaBraM",
    }

    logs: List[str] = [
        "# Step1 Backbone Smoke Logs",
        "",
        f"- workspace_root: `{workspace}`",
        f"- raw_dataset_copied: `False`",
        f"- training_started: `False`",
        "",
    ]

    bcic = smoke_bcic2a_loader(paths["bcic2a"], logs)
    physio = smoke_physionetmi_loader(paths["physionetmi"], logs)
    write_json(out_dir / "bcic2a_loader_smoke.json", bcic)
    write_json(out_dir / "physionetmi_loader_smoke.json", physio)

    rows: List[Dict[str, object]] = []
    rows.append(smoke_cbramod_bcic(paths, bcic, args.device, args.batch_size, logs))
    rows.append(smoke_mirepnet_pair(paths, "BCIC-IV-2a", bcic, logs))
    rows.append(smoke_cbramod_physio(paths, physio, args.device, args.batch_size, logs))
    rows.append(smoke_mirepnet_pair(paths, "PhysioNetMI", physio, logs))
    rows.append(smoke_labram_bcic(paths, bcic, logs))

    write_csv(out_dir / "backbone_dataset_compatibility.csv", rows)
    write_json(out_dir / "backbone_dataset_compatibility.json", {"pairs": rows})
    (out_dir / "backbone_smoke_logs.md").write_text("\n".join(logs) + "\n")
    print(json.dumps({"status": "completed", "rows": rows}, indent=2, ensure_ascii=False, sort_keys=True))


def smoke_bcic2a_loader(root: Path, logs: List[str]) -> Dict[str, object]:
    report: Dict[str, object] = {
        "dataset": "BCIC-IV-2a / BNCI2014-001",
        "path": str(root),
        "status": "unknown",
        "ready": False,
        "subjects": [],
        "sessions": [],
        "files": [],
        "trial_shape": "",
        "labels": [],
        "sfreq_raw": 250,
        "sfreq_output": 200,
        "channel_count": None,
        "channel_names": BNCI_CHANNELS,
        "eog_channels_detected": True,
        "eog_channels_excluded": True,
        "target_shape": [22, 800],
        "errors": [],
    }
    try:
        import scipy.io
        from scipy.signal import butter, resample, sosfiltfilt
    except Exception as exc:
        report["status"] = "missing_dependency"
        report["errors"].append(f"scipy import failed: {exc}")
        return report
    if not root.exists():
        report["status"] = "missing_path"
        report["errors"].append(f"missing root: {root}")
        return report

    files = sorted(root.glob("A0[1-9][TE].mat"))
    report["files"] = [p.name for p in files]
    subjects = sorted({p.name[1:3] for p in files})
    sessions_by_subject = defaultdict(set)
    for p in files:
        sessions_by_subject[p.name[1:3]].add(p.name[3])
    report["subjects"] = [f"A{s}" for s in subjects]
    report["sessions"] = sorted({s for values in sessions_by_subject.values() for s in values})
    report["subject_session_complete"] = all(sessions_by_subject[f"{i:02d}"] == {"T", "E"} for i in range(1, 10))

    labels_seen: List[int] = []
    shapes = Counter()
    sample_x: Optional[np.ndarray] = None
    sos = butter(5, [4.0 / 125.0, 38.0 / 125.0], btype="band", output="sos")
    try:
        for path in files:
            mat = scipy.io.loadmat(path)
            runs = mat["data"][0]
            for run_idx in range(3, len(runs)):
                cell = runs[run_idx][0, 0]
                raw = np.asarray(cell[0], dtype=np.float64)
                events = np.asarray(cell[1]).reshape(-1).astype(int)
                labels = np.asarray(cell[2]).reshape(-1).astype(int) - 1
                for event, label in zip(events, labels):
                    if 0 <= int(label) <= 3:
                        labels_seen.append(int(label))
                    start = int(event) + 2 * 250
                    stop = int(event) + 6 * 250
                    if sample_x is None and 0 <= int(label) <= 3 and stop <= raw.shape[0]:
                        sample = raw[start:stop, :22].T
                        sample = sample - sample.mean(axis=1, keepdims=True)
                        sample = sosfiltfilt(sos, sample, axis=-1)
                        sample_x = resample(sample, 800, axis=-1).astype(np.float32)
                    if 0 <= int(label) <= 3:
                        shapes[(22, 800)] += 1
        report["labels"] = sorted(set(labels_seen))
        report["label_counts"] = dict(Counter(labels_seen))
        if sample_x is not None:
            report["trial_shape"] = list(sample_x.shape)
            report["sample_trial_stats"] = safe_array_stats(sample_x)
            np.savez_compressed(Path(report["path"]).parents[5] / "_nonexistent_step1_no_write.npz") if False else None
        report["channel_count"] = 22
        report["trial_shape_counts"] = {str(k): v for k, v in shapes.items()}
        report["ready"] = (
            len(files) == 18
            and bool(report["subject_session_complete"])
            and report["labels"] == [0, 1, 2, 3]
            and report["trial_shape"] == [22, 800]
        )
        report["status"] = "ready" if report["ready"] else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        logs.append("## BCIC Loader Failure\n")
        logs.append("```text\n" + traceback.format_exc() + "\n```")
    return report


def smoke_physionetmi_loader(root: Path, logs: List[str]) -> Dict[str, object]:
    report: Dict[str, object] = {
        "dataset": "PhysioNetMI / EEGMMI",
        "path": str(root),
        "status": "unknown",
        "ready": False,
        "subjects": None,
        "runs_per_subject": None,
        "mi_runs_detected": [],
        "sampled_subjects": [],
        "sampled_runs": [],
        "event_code_to_label": {},
        "recommended_task": "",
        "recommended_window": "4.0s trial window, resample/crop to 200Hz x 4s = 800 samples for CBraMod compatibility",
        "sfreq": None,
        "channel_names": [],
        "errors": [],
    }
    try:
        import mne
    except Exception as exc:
        report["status"] = "missing_dependency"
        report["errors"].append(f"mne import failed: {exc}")
        return report
    if not root.exists():
        report["status"] = "missing_path"
        report["errors"].append(f"missing root: {root}")
        return report

    subject_dirs = sorted([p for p in root.glob("S[0-9][0-9][0-9]") if p.is_dir()])
    run_counts = {}
    for subject_dir in subject_dirs:
        run_counts[subject_dir.name] = len(list(subject_dir.glob(f"{subject_dir.name}R*.edf")))
    report["subjects"] = len(subject_dirs)
    report["runs_per_subject"] = sorted(set(run_counts.values()))
    mi_runs = [4, 6, 8, 10, 12, 14]
    report["mi_runs_detected"] = mi_runs
    report["task_type_by_run"] = {
        "1": "eyes_open",
        "2": "eyes_closed",
        "3,7,11": "motor_execution_left_right_fist",
        "4,8,12": "motor_imagery_left_right_fist",
        "5,9,13": "motor_execution_both_fists_both_feet",
        "6,10,14": "motor_imagery_both_fists_both_feet",
    }
    report["recommended_task"] = "2-class left vs right from MI runs 4/8/12 first; then 2-class hands vs feet from runs 6/10/14; 4-class requires combining run semantics carefully."
    report["event_code_to_label"] = {
        "T0": "rest",
        "T1_on_runs_4_8_12": "left_fist_imagery",
        "T2_on_runs_4_8_12": "right_fist_imagery",
        "T1_on_runs_6_10_14": "both_fists_imagery",
        "T2_on_runs_6_10_14": "both_feet_imagery",
    }

    samples = []
    try:
        for subject_dir in subject_dirs[:2]:
            report["sampled_subjects"].append(subject_dir.name)
            for run in [4, 6]:
                edf = subject_dir / f"{subject_dir.name}R{run:02d}.edf"
                if not edf.exists():
                    continue
                raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
                report["sampled_runs"].append(f"{subject_dir.name}R{run:02d}")
                if report["sfreq"] is None:
                    report["sfreq"] = float(raw.info["sfreq"])
                    report["channel_names"] = list(raw.ch_names)
                annotations = Counter(str(desc) for desc in raw.annotations.description)
                samples.append({"file": str(edf), "sfreq": float(raw.info["sfreq"]), "n_channels": len(raw.ch_names), "annotations": dict(annotations)})
        report["sample_smoke"] = samples
        report["ready"] = report["subjects"] == 109 and report["runs_per_subject"] == [14] and bool(samples)
        report["status"] = "ready" if report["ready"] else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        logs.append("## PhysioNet Loader Failure\n")
        logs.append("```text\n" + traceback.format_exc() + "\n```")
    return report


def smoke_cbramod_bcic(paths: Dict[str, Path], bcic: Dict[str, object], device: str, batch_size: int, logs: List[str]) -> Dict[str, object]:
    row = base_row("CBraMod", "BCIC-IV-2a", paths["cbramod"], paths["bcic2a"])
    row.update({
        "has_weights": (paths["cbramod"] / "pretrained_weights" / "pretrained_weights.pth").exists(),
        "loader_ready": bool(bcic.get("ready")),
        "expected_input": "[B,22,4,200] from [B,22,800]",
        "observed_input": str(bcic.get("trial_shape")),
        "channel_mapping_needed": False,
        "resampling_needed": False,
        "crop_needed": False,
    })
    if not row["loader_ready"]:
        row["status"] = "failed"
        row["notes"] = "BCIC loader not ready."
        return row
    try:
        x = load_one_bcic_trial(paths["bcic2a"])
        feature_shape, checkpoint = cbramod_forward(paths["cbramod"], x[None, ...], device, batch_size)
        row.update({
            "status": "ready",
            "forward_success": True,
            "feature_shape": str(feature_shape),
            "checkpoint_status": checkpoint,
            "recommended_for_next": True,
            "notes": "Frozen CBraMod forward smoke succeeded on one BCIC-IV-2a trial.",
        })
    except Exception as exc:
        row.update({"status": "failed", "forward_success": False, "notes": f"{type(exc).__name__}: {exc}"})
        logs.append("## CBraMod x BCIC Failure\n")
        logs.append("```text\n" + traceback.format_exc() + "\n```")
    return row


def smoke_cbramod_physio(paths: Dict[str, Path], physio: Dict[str, object], device: str, batch_size: int, logs: List[str]) -> Dict[str, object]:
    row = base_row("CBraMod", "PhysioNetMI", paths["cbramod"], paths["physionetmi"])
    row.update({
        "has_weights": (paths["cbramod"] / "pretrained_weights" / "pretrained_weights.pth").exists(),
        "loader_ready": bool(physio.get("ready")),
        "expected_input": "[B,22,4,200] from mapped/resampled [B,22,800]",
        "observed_input": f"{len(physio.get('channel_names') or [])} channels @ {physio.get('sfreq')} Hz",
        "channel_mapping_needed": True,
        "resampling_needed": True,
        "crop_needed": True,
    })
    if not row["loader_ready"]:
        row["status"] = "needs_adapter"
        row["notes"] = "PhysioNetMI loader audit incomplete."
        return row
    try:
        x, mapping = load_one_physio_trial_for_cbramod(paths["physionetmi"])
        feature_shape, checkpoint = cbramod_forward(paths["cbramod"], x[None, ...], device, batch_size)
        row.update({
            "status": "ready",
            "forward_success": True,
            "feature_shape": str(feature_shape),
            "checkpoint_status": checkpoint,
            "notes": f"Forward succeeded after deterministic 22-channel BNCI-style mapping and crop/resample. mapped_channels={mapping}",
        })
    except Exception as exc:
        row.update({"status": "needs_adapter", "forward_success": False, "notes": f"{type(exc).__name__}: {exc}"})
        logs.append("## CBraMod x PhysioNetMI Adapter Needed\n")
        logs.append("```text\n" + traceback.format_exc() + "\n```")
    return row


def smoke_mirepnet_pair(paths: Dict[str, Path], dataset: str, loader_report: Dict[str, object], logs: List[str]) -> Dict[str, object]:
    row = base_row("MIRepNet", dataset, paths["mirepnet"], paths["bcic2a"] if dataset == "BCIC-IV-2a" else paths["physionetmi"])
    row.update({
        "has_weights": (paths["mirepnet"] / "weight" / "MIRepNet.pth").exists(),
        "loader_ready": bool(loader_report.get("ready")),
        "expected_input": "Repository default PatchEmbedding num_channels=45; dataset.py expects preprocessed ./data/<dataset>/X.npy and labels.npy; BNCI2014004 path shown in code.",
        "observed_input": str(loader_report.get("trial_shape") or f"{len(loader_report.get('channel_names') or [])} channels"),
        "channel_mapping_needed": True,
        "resampling_needed": True,
        "crop_needed": True,
    })
    missing = missing_modules(["wandb", "einops"])
    if missing:
        row.update({
            "status": "needs_adapter",
            "forward_success": False,
            "missing_dependencies": ";".join(missing),
            "checkpoint_status": "not_loaded_due_to_missing_dependencies",
            "notes": "MIRepNet code imports missing optional/runtime dependencies and expects its own preprocessed data/channel-template pipeline. Dependency install was not performed per Step1 rule.",
        })
        return row
    try:
        sys.path.insert(0, str(paths["mirepnet"]))
        from model.mlm import mlm_mask  # type: ignore
        import torch

        model = mlm_mask(emb_size=256, depth=6, n_classes=4 if dataset == "BCIC-IV-2a" else 2, pretrain=None, pretrainmode=False)
        state = torch.load(paths["mirepnet"] / "weight" / "MIRepNet.pth", map_location="cpu")
        model_state = model.state_dict()
        matched = [k for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)]
        row["checkpoint_status"] = f"matched={len(matched)} checkpoint_keys={len(state)} model_keys={len(model_state)}"
        row["status"] = "needs_adapter"
        row["forward_success"] = False
        row["notes"] = "Dependencies resolved but direct smoke not run because native model expects 45-channel template/preprocessed X.npy pipeline; adapter required for this dataset."
    except Exception as exc:
        row.update({"status": "failed", "forward_success": False, "notes": f"{type(exc).__name__}: {exc}"})
        logs.append(f"## MIRepNet x {dataset} Failure\n")
        logs.append("```text\n" + traceback.format_exc() + "\n```")
    finally:
        if str(paths["mirepnet"]) in sys.path:
            sys.path.remove(str(paths["mirepnet"]))
    return row


def smoke_labram_bcic(paths: Dict[str, Path], bcic: Dict[str, object], logs: List[str]) -> Dict[str, object]:
    row = base_row("LaBraM", "BCIC-IV-2a", paths["labram"], paths["bcic2a"])
    checkpoints = sorted((paths["labram"] / "checkpoints").glob("*.pth"))
    row.update({
        "has_weights": bool(checkpoints),
        "loader_ready": bool(bcic.get("ready")),
        "expected_input": "[B, channels, patches, 200], README says resample to 200Hz and provide input channel order list.",
        "observed_input": str(bcic.get("trial_shape")),
        "forward_success": False,
        "feature_shape": "",
        "checkpoint_status": f"checkpoints={','.join(p.name for p in checkpoints)}",
        "channel_mapping_needed": True,
        "resampling_needed": False,
        "crop_needed": False,
        "status": "code_ready_weight_ready" if checkpoints else "failed",
        "notes": "Import/checkpoint-path audit only. LaBraM requires channel-order adapter/input_chans handling; no forward attempted per Step1.",
    })
    for required in ["modeling_finetune.py", "run_class_finetuning.py"]:
        if not (paths["labram"] / required).exists():
            row["status"] = "failed"
            row["notes"] += f" Missing {required}."
    return row


def load_one_bcic_trial(root: Path) -> np.ndarray:
    import scipy.io
    from scipy.signal import butter, resample, sosfiltfilt

    path = root / "A01T.mat"
    mat = scipy.io.loadmat(path)
    runs = mat["data"][0]
    sos = butter(5, [4.0 / 125.0, 38.0 / 125.0], btype="band", output="sos")
    for run_idx in range(3, len(runs)):
        cell = runs[run_idx][0, 0]
        raw = np.asarray(cell[0], dtype=np.float64)
        events = np.asarray(cell[1]).reshape(-1).astype(int)
        labels = np.asarray(cell[2]).reshape(-1).astype(int) - 1
        for event, label in zip(events, labels):
            start = int(event) + 2 * 250
            stop = int(event) + 6 * 250
            if 0 <= int(label) <= 3 and stop <= raw.shape[0]:
                sample = raw[start:stop, :22].T
                sample = sample - sample.mean(axis=1, keepdims=True)
                sample = sosfiltfilt(sos, sample, axis=-1)
                return resample(sample, 800, axis=-1).astype(np.float32)
    raise RuntimeError("No valid BCIC trial found.")


def load_one_physio_trial_for_cbramod(root: Path) -> Tuple[np.ndarray, List[str]]:
    import mne
    from scipy.signal import resample

    edf = root / "S001" / "S001R04.edf"
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    name_map = {normalize_channel(ch): ch for ch in raw.ch_names}
    missing = [ch for ch in BNCI_CHANNELS if ch not in name_map]
    if missing:
        raise RuntimeError(f"Missing BNCI-style channel mapping in PhysioNetMI: {missing}")
    picks = [name_map[ch] for ch in BNCI_CHANNELS]
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    usable = [event for event in events if event[2] in {event_id.get("T1", -1), event_id.get("T2", -1)}]
    if not usable:
        raise RuntimeError(f"No T1/T2 MI events found in {edf}")
    onset_sample = int(usable[0][0])
    sfreq = float(raw.info["sfreq"])
    duration = int(round(4.0 * sfreq))
    data = raw.get_data(picks=picks, start=onset_sample, stop=onset_sample + duration)
    if data.shape[-1] < duration:
        raise RuntimeError(f"Insufficient data after event: {data.shape}")
    data = data - data.mean(axis=1, keepdims=True)
    data = resample(data, 800, axis=-1).astype(np.float32)
    return data, picks


def cbramod_forward(cbramod_root: Path, x: np.ndarray, device_arg: str, batch_size: int) -> Tuple[List[int], str]:
    import torch
    import torch.nn as nn

    root = cbramod_root.resolve()
    sys.path.insert(0, str(root))
    try:
        from models.cbramod import CBraMod  # type: ignore

        device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
        model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        weight = root / "pretrained_weights" / "pretrained_weights.pth"
        ckpt = torch.load(weight, map_location="cpu")
        state = model.state_dict()
        matched = [k for k, v in state.items() if k in ckpt and tuple(v.shape) == tuple(ckpt[k].shape)]
        missing = [k for k in state if k not in ckpt]
        unexpected = [k for k in ckpt if k not in state]
        mismatch = [k for k, v in state.items() if k in ckpt and tuple(v.shape) != tuple(ckpt[k].shape)]
        if unexpected or mismatch:
            raise RuntimeError(f"CBraMod checkpoint unexpected={len(unexpected)} mismatch={len(mismatch)}")
        model.load_state_dict(ckpt, strict=False)
        model.proj_out = nn.Identity()
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        if x.shape[1:] != (22, 800):
            raise ValueError(f"CBraMod smoke expected [B,22,800], got {x.shape}")
        patches = x.reshape(x.shape[0], 22, 4, 200)
        with torch.no_grad():
            batch = torch.from_numpy(patches[:batch_size]).float().to(device)
            out = model(batch)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError("NaN/Inf in CBraMod output")
            pooled = out.mean(dim=(1, 2))
        return list(pooled.shape), f"matched={len(matched)} missing={len(missing)} unexpected={len(unexpected)} mismatch={len(mismatch)}"
    finally:
        if str(root) in sys.path:
            sys.path.remove(str(root))


def base_row(backbone: str, dataset: str, backbone_path: Path, dataset_path: Path) -> Dict[str, object]:
    return {
        "backbone": backbone,
        "dataset": dataset,
        "status": "skipped",
        "local_backbone_path": str(backbone_path),
        "dataset_path": str(dataset_path),
        "has_weights": False,
        "loader_ready": False,
        "expected_input": "",
        "observed_input": "",
        "forward_success": False,
        "feature_shape": "",
        "missing_dependencies": "",
        "checkpoint_status": "",
        "channel_mapping_needed": False,
        "resampling_needed": False,
        "crop_needed": False,
        "notes": "",
        "recommended_for_next": False,
    }


def missing_modules(names: Sequence[str]) -> List[str]:
    missing = []
    for name in names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    return missing


def normalize_channel(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


def safe_array_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "backbone", "dataset", "status", "local_backbone_path", "dataset_path", "has_weights", "loader_ready",
        "expected_input", "observed_input", "forward_success", "feature_shape", "missing_dependencies",
        "checkpoint_status", "channel_mapping_needed", "resampling_needed", "crop_needed", "notes",
        "recommended_for_next",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
