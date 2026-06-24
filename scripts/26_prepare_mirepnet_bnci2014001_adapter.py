#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt


WORKSPACE = Path("/ai/224duibishiyan/615新研究")
RAW_ROOT = Path("/ai/224duibishiyan/CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
OUT_DIR = WORKSPACE / "third_party/backbones/MIRepNet/data/BNCI2014001"
REPORT_DIR = WORKSPACE / "outputs/mirepnet_full_paper_code"


def iter_trials(path: Path):
    mat = scipy.io.loadmat(path)
    runs = mat["data"][0]
    for run_idx in range(3, len(runs)):
        cell = runs[run_idx][0, 0]
        raw = np.asarray(cell[0], dtype=np.float64)
        events = np.asarray(cell[1]).reshape(-1).astype(int)
        labels = np.asarray(cell[2]).reshape(-1).astype(int) - 1
        for event, label in zip(events, labels):
            label = int(label)
            if not 0 <= label <= 3:
                continue
            start = int(event) + 2 * 250
            stop = int(event) + 6 * 250
            if stop > raw.shape[0]:
                continue
            # First 22 columns are EEG channels; EOG channels are excluded.
            trial = raw[start:stop, :22].T
            yield trial, label


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_ROOT.glob("A0[1-9][TE].mat"))
    if len(files) != 18:
        raise SystemExit(f"Expected 18 BCIC IV-2a files, found {len(files)} in {RAW_ROOT}")

    sos = butter(5, [8.0 / 125.0, 30.0 / 125.0], btype="band", output="sos")

    xs = []
    ys = []
    subjects = []
    sessions = []
    trial_ids = []
    dropped = []
    counts = Counter()
    per_subject = Counter()
    per_subject_session = Counter()

    for path in files:
        subject = int(path.name[1:3]) - 1
        session = path.name[3]
        local_trial_id = 0
        for trial, label in iter_trials(path):
            trial = trial - trial.mean(axis=1, keepdims=True)
            trial = sosfiltfilt(sos, trial, axis=-1).astype(np.float32)
            if trial.shape != (22, 1000):
                dropped.append({"file": path.name, "reason": f"shape={trial.shape}"})
                continue
            if not np.isfinite(trial).all():
                dropped.append({"file": path.name, "reason": "non_finite"})
                continue
            xs.append(trial)
            ys.append(label)
            subjects.append(subject)
            sessions.append(session)
            trial_ids.append(f"{path.stem}:{local_trial_id}")
            local_trial_id += 1
            counts[label] += 1
            per_subject[subject] += 1
            per_subject_session[(subject, session)] += 1

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    subjects_arr = np.asarray(subjects, dtype=np.int64)
    sessions_arr = np.asarray(sessions)
    trial_ids_arr = np.asarray(trial_ids)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "labels.npy", y)
    np.save(OUT_DIR / "subjects.npy", subjects_arr)
    np.save(OUT_DIR / "sessions.npy", sessions_arr)
    np.save(OUT_DIR / "trial_ids.npy", trial_ids_arr)

    manifest = {
        "status": "completed",
        "protocol_label": "MIRepNet_BNCI2014001_4class_adapter",
        "raw_root": str(RAW_ROOT),
        "output_dir": str(OUT_DIR),
        "raw_dataset_copied": False,
        "saved_processed_arrays": ["X.npy", "labels.npy", "subjects.npy", "sessions.npy", "trial_ids.npy"],
        "x_shape": list(X.shape),
        "labels_shape": list(y.shape),
        "subjects_zero_based": sorted(np.unique(subjects_arr).astype(int).tolist()),
        "sessions": sorted(set(sessions_arr.tolist())),
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "per_subject_counts": {str(k): int(v) for k, v in sorted(per_subject.items())},
        "per_subject_session_counts": {f"S{k[0]}{k[1]}": int(v) for k, v in sorted(per_subject_session.items())},
        "trial_window": "event + 2s to event + 6s",
        "sfreq": 250,
        "bandpass": "8-30Hz butterworth order 5 sosfiltfilt",
        "channels_saved": 22,
        "saved_channel_order": [
            "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2",
            "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"
        ],
        "mi_repnet_training_behavior": "Official train_subject applies train/val split first, then EA and 45-channel inverse-distance interpolation inside process_and_replace_loader.",
        "dropped": dropped,
        "array_stats": {
            "mean": float(X.mean()),
            "std": float(X.std()),
            "max_abs": float(np.max(np.abs(X))),
            "nan_count": int(np.isnan(X).sum()),
            "inf_count": int(np.isinf(X).sum()),
        },
    }
    (REPORT_DIR / "mirepnet_bnci2014001_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
