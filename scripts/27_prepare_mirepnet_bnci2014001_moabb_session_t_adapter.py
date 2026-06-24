#!/usr/bin/env python3
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


WORKSPACE = Path("/ai/224duibishiyan/615新研究")
OUT_DIR = WORKSPACE / "third_party/backbones/MIRepNet/data/BNCI2014001"
REPORT_DIR = WORKSPACE / "outputs/mirepnet_full_paper_code"
BACKUP_ROOT = REPORT_DIR / "adapter_backups"

LABEL_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


def backup_existing_adapter() -> str:
    if not OUT_DIR.exists():
        return ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"BNCI2014001_before_moabb_session_t_{stamp}"
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OUT_DIR, backup_dir)
    return str(backup_dir)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_existing_adapter()

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30)

    xs = []
    ys = []
    subjects = []
    sessions = []
    trial_ids = []

    per_subject_counts = Counter()
    label_counts = Counter()
    session_counts = Counter()
    source_shapes = {}

    for subject in dataset.subject_list:
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
        source_shapes[str(subject)] = list(X.shape)
        session_mask = meta["session"].astype(str).eq("0train").to_numpy()
        X = X[session_mask]
        labels = labels[session_mask]
        meta = meta.loc[session_mask].reset_index(drop=True)

        if X.shape[0] != 288:
            raise RuntimeError(f"Subject {subject} session 0train expected 288 trials, got {X.shape[0]}")

        if X.shape[-1] < 1000:
            raise RuntimeError(f"Subject {subject} expected at least 1000 time points, got {X.shape[-1]}")

        X = X[:, :, :1000].astype(np.float32)
        y = np.asarray([LABEL_MAP[str(label)] for label in labels], dtype=np.int64)
        subject_zero_based = subject - 1

        xs.append(X)
        ys.append(y)
        subjects.append(np.full(X.shape[0], subject_zero_based, dtype=np.int64))
        sessions.append(np.asarray(["0train"] * X.shape[0]))
        trial_ids.append(np.asarray([f"S{subject:02d}:0train:{i:03d}" for i in range(X.shape[0])]))

        per_subject_counts[subject_zero_based] += X.shape[0]
        session_counts["0train"] += X.shape[0]
        label_counts.update(y.tolist())

    X_all = np.concatenate(xs, axis=0).astype(np.float32)
    y_all = np.concatenate(ys, axis=0).astype(np.int64)
    subjects_all = np.concatenate(subjects, axis=0).astype(np.int64)
    sessions_all = np.concatenate(sessions, axis=0)
    trial_ids_all = np.concatenate(trial_ids, axis=0)

    if X_all.shape != (2592, 22, 1000):
        raise RuntimeError(f"Expected adapter shape (2592, 22, 1000), got {X_all.shape}")
    if sorted(label_counts.values()) != [648, 648, 648, 648]:
        raise RuntimeError(f"Expected balanced 648 trials per class, got {dict(label_counts)}")

    np.save(OUT_DIR / "X.npy", X_all)
    np.save(OUT_DIR / "labels.npy", y_all)
    np.save(OUT_DIR / "subjects.npy", subjects_all)
    np.save(OUT_DIR / "sessions.npy", sessions_all)
    np.save(OUT_DIR / "trial_ids.npy", trial_ids_all)

    manifest = {
        "status": "completed",
        "protocol_label": "MIRepNet_BNCI2014001_4class_MOABB_session_T_only",
        "author_issue_alignment": {
            "BNCI2014001_4class_trials": 2592,
            "session_selection": "MOABB first session / 0train only",
            "train_test_behavior": "MIRepNet code performs subject-wise 30% train / 70% test split later, then EA and 45-channel interpolation separately.",
        },
        "output_dir": str(OUT_DIR),
        "previous_adapter_backup_dir": backup_dir,
        "raw_dataset_copied": False,
        "saved_processed_arrays": ["X.npy", "labels.npy", "subjects.npy", "sessions.npy", "trial_ids.npy"],
        "x_shape": list(X_all.shape),
        "labels_shape": list(y_all.shape),
        "source_moabb_shapes_before_session_filter": source_shapes,
        "time_crop": "MOABB returned 1001 samples for [2s, 6s]; saved first 1000 samples for MIRepNet's 4s@250Hz input.",
        "subjects_zero_based": sorted(np.unique(subjects_all).astype(int).tolist()),
        "sessions": sorted(set(sessions_all.tolist())),
        "label_map": LABEL_MAP,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "per_subject_counts": {str(k): int(v) for k, v in sorted(per_subject_counts.items())},
        "session_counts": {str(k): int(v) for k, v in sorted(session_counts.items())},
        "sfreq": 250,
        "bandpass": "MOABB MotorImagery(fmin=8, fmax=30)",
        "channels_saved": 22,
        "saved_channel_order": [
            "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2",
            "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"
        ],
        "array_stats": {
            "mean": float(X_all.mean()),
            "std": float(X_all.std()),
            "max_abs": float(np.max(np.abs(X_all))),
            "nan_count": int(np.isnan(X_all).sum()),
            "inf_count": int(np.isinf(X_all).sum()),
        },
    }
    out_path = REPORT_DIR / "mirepnet_bnci2014001_moabb_session_t_adapter_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
