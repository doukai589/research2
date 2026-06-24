from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
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


def load_bcic2a_trials(data_root: str, subjects: Optional[Sequence[int]] = None) -> List[TrialRecord]:
    try:
        import scipy.io
        from scipy.signal import butter, resample, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError("BCIC2a loading requires scipy.") from exc

    root = Path(data_root)
    subject_set = set(subjects or range(1, 10))
    files = []
    for path in sorted(root.glob("A0*.mat")):
        match = SUBJECT_FILE_RE.match(path.name)
        if match and int(match.group("subject")) in subject_set:
            files.append((int(match.group("subject")), match.group("session"), path))
    if not files:
        raise FileNotFoundError(f"No A0*.mat files found in {data_root}")

    sos = butter(5, [4.0 / 125.0, 38.0 / 125.0], btype="band", output="sos")
    records: List[TrialRecord] = []
    for subject, session, path in files:
        mat = scipy.io.loadmat(path)
        runs = mat["data"][0]
        for run_idx in range(3, len(runs)):
            cell = runs[run_idx][0, 0]
            raw = np.asarray(cell[0], dtype=np.float64)[:, :22]
            events = np.asarray(cell[1]).reshape(-1).astype(int)
            labels = np.asarray(cell[2]).reshape(-1).astype(int) - 1
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
                records.append(
                    TrialRecord(
                        x=sample,
                        y=int(label),
                        subject=subject,
                        session=session,
                        trial_id=f"A{subject:02d}{session}_run{run_idx:02d}_trial{trial_idx:03d}",
                    )
                )
    return records


def records_to_arrays(records: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot convert empty records to arrays.")
    return (
        np.stack([r.x for r in records]).astype(np.float32),
        np.asarray([r.y for r in records], dtype=np.int64),
        np.asarray([r.subject for r in records], dtype=np.int64),
        np.asarray([r.session for r in records]),
        np.asarray([r.trial_id for r in records]),
    )


def make_few_shot_split(records: Sequence[TrialRecord], target_subject: int, shot: int, seed: int):
    rng = np.random.default_rng(seed)
    source = [r for r in records if r.subject != target_subject]
    target_train = [r for r in records if r.subject == target_subject and r.session == "T"]
    target_test = [r for r in records if r.subject == target_subject and r.session == "E"]
    support: List[TrialRecord] = []
    for label in range(4):
        candidates = [r for r in target_train if r.y == label]
        if len(candidates) < shot:
            raise ValueError(f"Subject {target_subject} class {label} has {len(candidates)} trials, need {shot}.")
        idx = rng.choice(len(candidates), size=shot, replace=False)
        support.extend(candidates[int(i)] for i in idx)
    return source, support, target_test


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=(0, 2), keepdims=True), x.std(axis=(0, 2), keepdims=True))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

