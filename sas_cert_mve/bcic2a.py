from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


SUBJECT_FILE_RE = re.compile(r"A(?P<subject>\d{2})(?P<session>[TE])\.mat$")


@dataclass(frozen=True)
class TrialRecord:
    x: np.ndarray
    y: int
    subject: int
    session: str
    trial_id: str


def parse_subjects(value: str) -> List[int]:
    if value.strip().lower() == "all":
        return list(range(1, 10))
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def load_bcic2a_trials(
    data_root: str,
    subjects: Optional[Sequence[int]] = None,
    sessions: Sequence[str] = ("T", "E"),
    cache_npz: Optional[str] = None,
    force_rebuild: bool = False,
    low_hz: float = 4.0,
    high_hz: float = 38.0,
) -> List[TrialRecord]:
    """Load BCIC IV-2a MATLAB files into trial records.

    Samples are extracted as 2s-6s post-cue windows, band-pass filtered, resampled
    from 250 Hz to 200 Hz, and returned as `[22, 800]` float32 arrays.
    """

    if cache_npz and os.path.exists(cache_npz) and not force_rebuild:
        return _load_cache(cache_npz)

    try:
        import scipy.io
        from scipy.signal import butter, resample, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError(
            "BCIC2a loading requires scipy. Install sas_cert_mve/requirements.txt."
        ) from exc

    subjects_set = set(subjects or range(1, 10))
    sessions_set = set(sessions)
    files = []
    for name in sorted(os.listdir(data_root)):
        match = SUBJECT_FILE_RE.match(name)
        if not match:
            continue
        subject = int(match.group("subject"))
        session = match.group("session")
        if subject in subjects_set and session in sessions_set:
            files.append((subject, session, os.path.join(data_root, name)))

    if not files:
        raise FileNotFoundError(f"No AxxT/AxxE .mat files found in {data_root}")

    sos = butter(5, [low_hz / 125.0, high_hz / 125.0], btype="band", output="sos")
    records: List[TrialRecord] = []

    for subject, session, path in files:
        mat = scipy.io.loadmat(path)
        if "data" not in mat:
            raise KeyError(f"{path} does not contain MATLAB key 'data'")
        runs = mat["data"][0]
        for run_idx in range(3, len(runs)):
            cell = runs[run_idx][0, 0]
            raw = np.asarray(cell[0], dtype=np.float64)[:, :22]
            events = np.asarray(cell[1]).reshape(-1).astype(int)
            labels = np.asarray(cell[2]).reshape(-1)
            if len(events) == 0 or len(labels) == 0:
                continue
            labels = labels.astype(int) - 1
            for trial_idx, (event, label) in enumerate(zip(events, labels)):
                if label < 0 or label > 3:
                    continue
                start = int(event) + 2 * 250
                stop = int(event) + 6 * 250
                if start < 0 or stop > raw.shape[0]:
                    continue
                sample = raw[start:stop].T
                sample = sample - sample.mean(axis=1, keepdims=True)
                sample = sosfiltfilt(sos, sample, axis=-1)
                sample = resample(sample, 800, axis=-1).astype(np.float32)
                trial_id = f"A{subject:02d}{session}_run{run_idx:02d}_trial{trial_idx:03d}"
                records.append(TrialRecord(sample, int(label), subject, session, trial_id))

    if cache_npz:
        os.makedirs(os.path.dirname(os.path.abspath(cache_npz)), exist_ok=True)
        _save_cache(cache_npz, records)

    return records


def _save_cache(path: str, records: Sequence[TrialRecord]) -> None:
    np.savez_compressed(
        path,
        x=np.stack([r.x for r in records]).astype(np.float32),
        y=np.asarray([r.y for r in records], dtype=np.int64),
        subject=np.asarray([r.subject for r in records], dtype=np.int64),
        session=np.asarray([r.session for r in records]),
        trial_id=np.asarray([r.trial_id for r in records]),
    )


def _load_cache(path: str) -> List[TrialRecord]:
    data = np.load(path, allow_pickle=False)
    return [
        TrialRecord(
            x=data["x"][i].astype(np.float32),
            y=int(data["y"][i]),
            subject=int(data["subject"][i]),
            session=str(data["session"][i]),
            trial_id=str(data["trial_id"][i]),
        )
        for i in range(data["x"].shape[0])
    ]


def records_to_arrays(records: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([r.x for r in records]).astype(np.float32),
        np.asarray([r.y for r in records], dtype=np.int64),
        np.asarray([r.subject for r in records], dtype=np.int64),
        np.asarray([r.session for r in records]),
        np.asarray([r.trial_id for r in records]),
    )


def select_records(
    records: Sequence[TrialRecord],
    subjects: Optional[Iterable[int]] = None,
    sessions: Optional[Iterable[str]] = None,
) -> List[TrialRecord]:
    subject_set = set(subjects) if subjects is not None else None
    session_set = set(sessions) if sessions is not None else None
    out = []
    for record in records:
        if subject_set is not None and record.subject not in subject_set:
            continue
        if session_set is not None and record.session not in session_set:
            continue
        out.append(record)
    return out


def make_few_shot_split(
    records: Sequence[TrialRecord],
    target_subject: int,
    shot: int,
    seed: int,
) -> Tuple[List[TrialRecord], List[TrialRecord], List[TrialRecord]]:
    """Return source records, target support records, and target test records."""

    rng = np.random.default_rng(seed)
    source = [r for r in records if r.subject != target_subject]
    target_train = [r for r in records if r.subject == target_subject and r.session == "T"]
    target_test = [r for r in records if r.subject == target_subject and r.session == "E"]
    if not target_test:
        target_test = [r for r in target_train]

    support: List[TrialRecord] = []
    for label in range(4):
        candidates = [r for r in target_train if r.y == label]
        if len(candidates) < shot:
            raise ValueError(
                f"Subject {target_subject} label {label} has {len(candidates)} train trials, "
                f"cannot sample {shot}-shot support."
            )
        picked = rng.choice(len(candidates), size=shot, replace=False)
        support.extend(candidates[int(i)] for i in picked)

    support_ids = {r.trial_id for r in support}
    target_test = [r for r in target_test if r.trial_id not in support_ids]
    return source, support, target_test


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=(0, 2), keepdims=True)
        std = x.std(axis=(0, 2), keepdims=True)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

