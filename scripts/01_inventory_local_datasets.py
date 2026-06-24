from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


WORKSPACE = Path.cwd().resolve()
OUT_DIR = WORKSPACE / "outputs" / "setup_audit"
EXTS = {".mat", ".gdf", ".edf", ".event", ".set", ".cnt", ".fif", ".vhdr", ".eeg", ".vmrk", ".npz", ".npy", ".pkl", ".pickle", ".h5", ".hdf5", ".csv", ".tsv"}
EXCLUDE_PARTS = {".git", "site-packages", "node_modules", "__pycache__", "proc", "sys", "dev", "conda", "envs"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roots = candidate_roots()
    log = ["# Dataset Search Log", "", f"- workspace: `{WORKSPACE}`", f"- candidate_roots: `{len(roots)}`", ""]
    files = []
    for root in roots:
        files.extend(scan_files(root))
    files = sorted(set(files))
    groups = infer_groups(files)
    rows = [summarize_group(name, paths) for name, paths in sorted(groups.items())]
    rows.sort(key=lambda r: (score_priority_order(r["priority"]), -int(r["mi_priority_score"]), r["inferred_dataset_name"]))
    write_csv(OUT_DIR / "dataset_inventory.csv", rows)
    payload = {"workspace_root": str(WORKSPACE), "scanned_roots": [str(r) for r in roots], "datasets": rows, "recommended_dataset_order": recommended_order(rows)}
    (OUT_DIR / "dataset_inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / "dataset_search_log.md").write_text("\n".join(log + search_log_lines(rows)) + "\n")
    write_mi_candidates(OUT_DIR / "mi_dataset_candidates.md", rows)
    print(json.dumps({"status": "completed", "dataset_count": len(rows), "recommended_dataset_order": payload["recommended_dataset_order"]}, indent=2, sort_keys=True))


def candidate_roots() -> List[Path]:
    roots = {WORKSPACE, WORKSPACE.parent}
    keywords = ["新研究", "CBraMod-main", "sas_cert_cbramod_mve", "data", "dataset", "datasets", "tmp_in", "MNE-bnci-data", "MOABB", "BCIC", "BCI", "OpenBMI", "SEED", "DEAP"]
    for base in [WORKSPACE, WORKSPACE.parent]:
        for depth1 in base.iterdir() if base.exists() else []:
            if not depth1.is_dir() or should_skip(depth1):
                continue
            if any(k.lower() in depth1.name.lower() for k in keywords):
                roots.add(depth1)
            for depth2 in depth1.iterdir() if depth1.is_dir() else []:
                if depth2.is_dir() and not should_skip(depth2) and any(k.lower() in str(depth2).lower() for k in keywords):
                    roots.add(depth2)
    return sorted(roots)


def should_skip(path: Path) -> bool:
    text = str(path)
    return any(part in path.parts for part in EXCLUDE_PARTS) or "/." in text


def scan_files(root: Path, max_depth: int = 6) -> List[Path]:
    out = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_depth = len(current.relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PARTS and not d.startswith(".")]
        if rel_depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            p = current / name
            if p.suffix.lower() in EXTS and not should_skip(p):
                out.append(p)
    return sorted(set(out))


def infer_groups(files: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        key = infer_dataset_key(p)
        groups[key].append(p)
    return groups


def infer_dataset_key(path: Path) -> str:
    s = str(path).lower()
    name = path.name
    if "001-2014" in s or "bnci2014001" in s or re.match(r"A0[1-9][TE]\.mat$", name):
        return "BCIC-IV-2a / BNCI2014-001"
    if "004-2014" in s or "bnci2014004" in s or "bcic2b" in s or re.match(r"B0[1-9].*\\.mat$", name):
        return "BCIC-IV-2b / BNCI2014-004"
    if "openbmi" in s or "lee2019" in s:
        return "OpenBMI / Lee2019"
    if "physionet" in s or "eegmmi" in s or "/tmp_in/mi/files/" in s or re.search(r"/files/s\d{3}/s\d{3}r\d{2}\.edf$", s):
        return "PhysioNetMI / EEGMMI"
    if "cho2017" in s:
        return "Cho2017 MI"
    if "zhou2016" in s:
        return "Zhou2016 MI"
    if "seed" in s:
        return "SEED / SEED-IV"
    if "deap" in s:
        return "DEAP"
    if "chb" in s:
        return "CHB-MIT"
    if "sleep" in s or "isruc" in s:
        return "Sleep / ISRUC"
    if "ssvep" in s:
        return "SSVEP"
    if "p300" in s or "erp" in s:
        return "P300 / ERP"
    return f"unknown::{common_parent_label(path)}"


def common_parent_label(path: Path) -> str:
    parts = path.parts
    for token in ["data", "dataset", "datasets", "tmp_in", "database"]:
        if token in [p.lower() for p in parts]:
            idx = [p.lower() for p in parts].index(token)
            return "/".join(parts[idx : min(len(parts), idx + 4)])
    return path.parent.name


def summarize_group(name: str, paths: Sequence[Path]) -> Dict[str, object]:
    paths = canonical_paths_for(name, paths)
    ext_counter = Counter(p.suffix.lower() for p in paths)
    total_size = sum(safe_size(p) for p in paths)
    root = common_root(paths)
    files = [p.name for p in paths]
    expected, found = expected_files(name, files)
    task_type = task_type_for(name)
    mi_score = mi_priority_score(name, found, expected)
    priority = priority_for(task_type, mi_score)
    return {
        "inferred_dataset_name": name,
        "task_type": task_type,
        "priority": priority,
        "root_path": str(root),
        "file_count": len(paths),
        "total_size_gb": round(total_size / (1024**3), 6),
        "extensions": ";".join(f"{k}:{v}" for k, v in sorted(ext_counter.items())),
        "subject_count_guess": subject_guess(files),
        "session_count_guess": session_guess(files),
        "expected_subject_files": ",".join(expected),
        "found_expected_files": ",".join(found),
        "has_train_eval_split_guess": has_train_eval(files),
        "format": ";".join(sorted(ext_counter)),
        "likely_loader": likely_loader(name),
        "do_not_copy_confirmed": True,
        "notes": notes_for(name, files, expected, found),
        "status": status_for(name, expected, found, len(paths)),
        "mi_priority_score": mi_score,
        "usability_score": usability_score(name, expected, found, paths),
        "existing_loader_compatibility_score": loader_score(name),
    }


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def canonical_paths_for(name: str, paths: Sequence[Path]) -> List[Path]:
    if "PhysioNetMI" not in name:
        return list(paths)
    preferred_roots = [
        WORKSPACE.parent / "CBraMod-main" / "tmp_in" / "MI" / "files",
        WORKSPACE.parent / "新研究" / "CBraMod-main" / "tmp_in" / "MI" / "files",
        WORKSPACE.parent / "EEG" / "tmp_in" / "MI" / "files",
    ]
    for root in preferred_roots:
        if not root.exists():
            continue
        subset = [p for p in paths if is_under(p, root)]
        if subset:
            return sorted(subset)
    raw = [p for p in paths if "/tmp_in/MI/files/" in str(p)]
    return sorted(raw or paths)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def common_root(paths: Sequence[Path]) -> Path:
    if not paths:
        return WORKSPACE
    return Path(os.path.commonpath([str(p.parent) for p in paths]))


def expected_files(name: str, files: Sequence[str]) -> tuple[List[str], List[str]]:
    if "BCIC-IV-2a" in name:
        expected = [f"A{i:02d}{s}.mat" for i in range(1, 10) for s in ["T", "E"]]
        found = [f for f in expected if f in set(files)]
        return expected, found
    if "BCIC-IV-2b" in name:
        expected = []
        found = [f for f in files if f.lower().endswith((".mat", ".gdf"))]
        return expected, found
    if "PhysioNetMI" in name:
        edf = sum(1 for f in files if re.match(r"S\d{3}R\d{2}\.edf$", f, flags=re.I))
        event = sum(1 for f in files if re.match(r"S\d{3}R\d{2}\.edf\.event$", f, flags=re.I))
        expected = ["109_subjects_x_14_runs_edf", "109_subjects_x_14_runs_event"]
        found = []
        if edf >= 1526:
            found.append(f"{edf}_edf")
        if event >= 1526:
            found.append(f"{event}_event")
        return expected, found
    return [], []


def task_type_for(name: str) -> str:
    low = name.lower()
    if "chb" in low:
        return "seizure"
    if any(k in low for k in ["bcic", "openbmi", "physionetmi", "eegmmi", "cho2017", "zhou2016", "motorimagery", "motor imagery"]):
        return "MI"
    if re.search(r"(^|[^a-z])mi([^a-z]|$)", low):
        return "MI"
    if "seed" in low or "deap" in low:
        return "emotion"
    if "sleep" in low or "isruc" in low:
        return "sleep"
    if "ssvep" in low:
        return "ssvep"
    if "p300" in low or "erp" in low:
        return "p300"
    return "unknown"


def mi_priority_score(name: str, found: Sequence[str], expected: Sequence[str]) -> int:
    low = name.lower()
    if "bcic-iv-2a" in low:
        return 100 if len(found) == 18 else 80
    if "bcic-iv-2b" in low:
        return 90
    if "openbmi" in low or "lee2019" in low:
        return 85
    if "cho2017" in low or "physionetmi" in low or "eegmmi" in low:
        return 75
    if task_type_for(name) == "MI":
        return 60
    if task_type_for(name) != "unknown":
        return 30
    return 10


def priority_for(task_type: str, score: int) -> str:
    if task_type == "MI" and score >= 85:
        return "primary"
    if task_type == "MI":
        return "secondary"
    if task_type != "unknown":
        return "optional"
    return "exclude"


def subject_guess(files: Sequence[str]) -> int:
    subjects = set()
    for f in files:
        for pat in [r"A(\d{2})[TE]", r"S(\d+)", r"subject[_-]?(\d+)", r"sub[_-]?(\d+)"]:
            m = re.search(pat, f, flags=re.I)
            if m:
                subjects.add(m.group(1))
    return len(subjects)


def session_guess(files: Sequence[str]) -> int:
    sessions = set()
    for f in files:
        if re.search(r"[TE]\.mat$", f):
            sessions.add(f[-5])
        m = re.search(r"S\d{3}R(\d{2})\.edf", f, flags=re.I)
        if m:
            sessions.add(m.group(1))
        for pat in [r"session[_-]?(\d+)", r"sess[_-]?(\d+)"]:
            m = re.search(pat, f, flags=re.I)
            if m:
                sessions.add(m.group(1))
    return len(sessions)


def has_train_eval(files: Sequence[str]) -> bool:
    text = " ".join(files).lower()
    return ("train" in text and ("eval" in text or "test" in text)) or bool(any(re.search(r"A0[1-9][TE]\.mat$", f) for f in files))


def likely_loader(name: str) -> str:
    if "BCIC-IV-2a" in name:
        return "existing sas_cert BCIC2a raw/MOABB loader"
    if "BCIC-IV-2b" in name:
        return "MOABB BNCI2014_004 or custom raw loader"
    if "OpenBMI" in name:
        return "MOABB Lee2019/OpenBMI if installed"
    if "PhysioNetMI" in name:
        return "MNE EEGBCI / MOABB PhysionetMI"
    return "manual inspection required"


def notes_for(name: str, files: Sequence[str], expected: Sequence[str], found: Sequence[str]) -> str:
    if "PhysioNetMI" in name:
        return f"PhysioNet EEGMMI raw MI root; expected 109 subjects x 14 runs. Found markers: {', '.join(found) or 'none'}."
    if expected:
        return f"Expected {len(expected)} files; found {len(found)}. Patterns: {', '.join(found[:6])}"
    return f"Recorded file name patterns only; no data copied. Example files: {', '.join(files[:5])}"


def status_for(name: str, expected: Sequence[str], found: Sequence[str], file_count: int) -> str:
    if "BCIC-IV-2a" in name and len(found) == 18:
        return "ready"
    if "PhysioNetMI" in name and any(str(item).endswith("_edf") for item in found):
        return "ready"
    if expected and len(found) < len(expected):
        return "incomplete"
    if file_count > 0 and task_type_for(name) != "unknown":
        return "likely_ready"
    if file_count > 0:
        return "unknown"
    return "exclude"


def usability_score(name: str, expected: Sequence[str], found: Sequence[str], paths: Sequence[Path]) -> int:
    score = 20
    if expected and len(found) == len(expected):
        score += 60
    if any(p.suffix.lower() in {".mat", ".gdf", ".edf"} for p in paths):
        score += 10
    if task_type_for(name) == "MI":
        score += 10
    return min(100, score)


def loader_score(name: str) -> int:
    if "BCIC-IV-2a" in name:
        return 100
    if "BCIC-IV-2b" in name:
        return 80
    if "OpenBMI" in name or "PhysioNetMI" in name:
        return 70
    return 30


def score_priority_order(priority: str) -> int:
    return {"primary": 0, "secondary": 1, "optional": 2, "exclude": 3}.get(priority, 9)


def recommended_order(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mi = [r for r in rows if r["task_type"] == "MI" and r["priority"] != "exclude"]
    mi = sorted(mi, key=lambda r: (-int(r["mi_priority_score"]), -int(r["usability_score"])))
    non_mi = [r["inferred_dataset_name"] for r in rows if r["task_type"] != "MI" and r["priority"] != "exclude"]
    return {
        "first": mi[0] if len(mi) > 0 else None,
        "second": mi[1] if len(mi) > 1 else None,
        "third": mi[2] if len(mi) > 2 else None,
        "not_recommended": [r["inferred_dataset_name"] for r in rows if r["priority"] == "exclude"],
        "non_mi_candidates": non_mi[:20],
    }


def search_log_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    lines = ["## Datasets Found", ""]
    for row in rows:
        lines.append(f"- {row['inferred_dataset_name']}: `{row['root_path']}` files={row['file_count']} status={row['status']}")
    return lines


def write_mi_candidates(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    mi = [r for r in rows if r["task_type"] == "MI" and r["priority"] != "exclude"]
    lines = ["# MI Dataset Candidates", ""]
    for i, row in enumerate(sorted(mi, key=lambda r: -int(r["mi_priority_score"])), 1):
        lines.append(f"{i}. **{row['inferred_dataset_name']}**")
        lines.append(f"   - path: `{row['root_path']}`")
        lines.append(f"   - files: `{row['file_count']}` format: `{row['format']}` status: `{row['status']}`")
        lines.append(f"   - score: `{row['mi_priority_score']}` notes: {row['notes']}")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
