"""PhysioNetMI / EEGMMI left-vs-right motor imagery loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    import mne
except Exception as exc:  # pragma: no cover
    mne = None
    MNE_IMPORT_ERROR = exc
else:
    MNE_IMPORT_ERROR = None

from sas_core.utils.io import write_json


TASK_LABELS = {"T1": 0, "T2": 1}


@dataclass(frozen=True)
class PhysioNetMIProtocol:
    root: Path
    runs: tuple[int, ...] = (4, 8, 12)
    train_subjects: tuple[int, ...] = tuple(range(1, 71))
    val_subjects: tuple[int, ...] = tuple(range(71, 90))
    test_subjects: tuple[int, ...] = tuple(range(90, 110))
    sfreq: int = 160
    window_seconds: int = 4
    bandpass_hz: tuple[float, float] = (1.0, 40.0)
    per_trial_channel_zscore: bool = True

    @property
    def samples(self) -> int:
        return int(self.sfreq * self.window_seconds)


def default_physionet_mi_protocol(root: str | Path = "/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files") -> PhysioNetMIProtocol:
    return PhysioNetMIProtocol(root=Path(root))


def _require_mne() -> None:
    if mne is None:
        raise RuntimeError(f"mne import failed: {MNE_IMPORT_ERROR}")


def build_physionet_mi_cache(
    cache_path: str | Path,
    protocol: PhysioNetMIProtocol | None = None,
    rebuild: bool = False,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build or load a derived PhysioNetMI trial-window cache.

    The cache contains processed trial windows, labels, subjects, runs, and channel
    names. It does not copy raw EDF files.
    """

    cache_path = Path(cache_path)
    protocol = protocol or default_physionet_mi_protocol()
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "X": data["X"],
            "y": data["y"],
            "subjects": data["subjects"],
            "runs": data["runs"],
            "channels": data["channels"].tolist(),
            "cache_path": str(cache_path),
            "rebuilt": False,
        }

    _require_mne()
    if not protocol.root.exists():
        raise FileNotFoundError(f"PhysioNetMI root not found: {protocol.root}")

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    subject_list: list[int] = []
    run_list: list[int] = []
    channel_names: list[str] | None = None
    sos = butter(4, list(protocol.bandpass_hz), btype="bandpass", fs=protocol.sfreq, output="sos")

    for subject_dir in sorted(protocol.root.glob("S[0-9][0-9][0-9]")):
        subject = int(subject_dir.name[1:])
        for run in protocol.runs:
            edf_path = subject_dir / f"S{subject:03d}R{run:02d}.edf"
            if not edf_path.exists():
                continue
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if abs(float(raw.info["sfreq"]) - protocol.sfreq) > 1e-6:
                raw.resample(protocol.sfreq, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if channel_names is None:
                channel_names = [raw.ch_names[i] for i in picks]
            eeg = raw.get_data(picks=picks).astype(np.float32)
            eeg = sosfiltfilt(sos, eeg, axis=1).astype(np.float32)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            event_name = {value: key for key, value in event_id.items()}
            for sample, _, code in events:
                desc = event_name.get(int(code))
                if desc not in TASK_LABELS:
                    continue
                start = int(sample)
                stop = start + protocol.samples
                if stop > eeg.shape[1]:
                    continue
                segment = eeg[:, start:stop].copy()
                if protocol.per_trial_channel_zscore:
                    segment -= segment.mean(axis=1, keepdims=True)
                    segment /= segment.std(axis=1, keepdims=True) + 1e-6
                x_list.append(segment.astype(np.float32))
                y_list.append(TASK_LABELS[desc])
                subject_list.append(subject)
                run_list.append(run)

    if not x_list:
        raise RuntimeError(f"No trials extracted from {protocol.root}")

    X = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subject_list, dtype=np.int64)
    runs = np.asarray(run_list, dtype=np.int64)
    channels = np.asarray(channel_names or [], dtype=object)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, subjects=subjects, runs=runs, channels=channels)

    if manifest_path is not None:
        write_json(manifest_path, make_physionet_mi_manifest(protocol, cache_path, X, y, subjects, channels))

    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "runs": runs,
        "channels": channels.tolist(),
        "cache_path": str(cache_path),
        "rebuilt": True,
    }


def make_physionet_mi_manifest(
    protocol: PhysioNetMIProtocol,
    cache_path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    channels: Sequence[str],
) -> Dict[str, Any]:
    split_counts = {}
    for split_name, split_subjects in (
        ("train", protocol.train_subjects),
        ("val", protocol.val_subjects),
        ("test", protocol.test_subjects),
    ):
        mask = np.isin(subjects, split_subjects)
        labels, counts = np.unique(y[mask], return_counts=True)
        split_counts[split_name] = {
            "n_trials": int(mask.sum()),
            "subjects": [int(x) for x in sorted(set(subjects[mask].tolist()))],
            "label_counts": {str(k): int(v) for k, v in zip(labels, counts)},
        }
    return {
        "dataset": "PhysioNetMI",
        "source_root": str(protocol.root),
        "raw_data_copied": False,
        "cache_path": str(cache_path),
        "cache_contains_raw_edf": False,
        "cache_contains_derived_trial_windows": True,
        "task": "left/right motor imagery, runs R04/R08/R12, T1/T2 only",
        "runs": list(protocol.runs),
        "bandpass_hz": list(protocol.bandpass_hz),
        "per_trial_channel_zscore": protocol.per_trial_channel_zscore,
        "window_seconds": protocol.window_seconds,
        "canonical_sfreq": protocol.sfreq,
        "canonical_shape": list(X.shape),
        "channels": [str(c) for c in channels],
        "split": {
            "train_subjects": [min(protocol.train_subjects), max(protocol.train_subjects)],
            "val_subjects": [min(protocol.val_subjects), max(protocol.val_subjects)],
            "test_subjects": [min(protocol.test_subjects), max(protocol.test_subjects)],
            "counts": split_counts,
        },
    }


def split_indices(subjects: np.ndarray, protocol: PhysioNetMIProtocol | None = None) -> Dict[str, np.ndarray]:
    protocol = protocol or default_physionet_mi_protocol()
    return {
        "train": np.where(np.isin(subjects, protocol.train_subjects))[0],
        "val": np.where(np.isin(subjects, protocol.val_subjects))[0],
        "test": np.where(np.isin(subjects, protocol.test_subjects))[0],
    }


def support_test_split_for_subject(
    y: np.ndarray,
    subjects: np.ndarray,
    target_subject: int,
    shot_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-subject support and held-out indices."""

    rng = np.random.default_rng(seed)
    target_indices = np.where(subjects == target_subject)[0]
    support: list[int] = []
    for label in sorted(np.unique(y[target_indices]).tolist()):
        candidates = target_indices[y[target_indices] == label]
        if len(candidates) < shot_per_class:
            raise ValueError(f"subject {target_subject} label {label} has only {len(candidates)} trials")
        support.extend(rng.choice(candidates, size=shot_per_class, replace=False).tolist())
    support_idx = np.asarray(sorted(support), dtype=np.int64)
    test_idx = np.asarray([i for i in target_indices.tolist() if i not in set(support_idx.tolist())], dtype=np.int64)
    return support_idx, test_idx

