from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


WORKSPACE = Path.cwd().resolve()
OUT_DIR = WORKSPACE / "outputs" / "setup_audit"
BACKBONE_ROOT = WORKSPACE / "third_party" / "backbones"


BACKBONES = [
    {
        "model_name": "CBraMod",
        "priority": "existing_primary",
        "repo_url": "https://github.com/wjq-learning/CBraMod",
        "local_path": None,
        "clone": False,
        "required": ["models/cbramod.py", "quick_example.py", "requirements.txt"],
        "weight": "pretrained_weights/pretrained_weights.pth",
    },
    {
        "model_name": "MIRepNet",
        "priority": "highest_mi_foundation",
        "repo_url": "https://github.com/staraink/MIRepNet.git",
        "local_path": BACKBONE_ROOT / "MIRepNet",
        "clone": True,
        "required": ["dataset.py", "finetune.py", "model"],
        "weight": "weight/MIRepNet.pth",
    },
    {
        "model_name": "LaBraM",
        "priority": "mid_high_eeg_foundation",
        "repo_url": "https://github.com/935963004/LaBraM.git",
        "local_path": BACKBONE_ROOT / "LaBraM",
        "clone": True,
        "required": ["run_class_finetuning.py"],
        "weight": "",
    },
    {
        "model_name": "EEGPT",
        "priority": "mid_high_eeg_foundation",
        "repo_url": "https://github.com/BINE022/EEGPT.git",
        "local_path": BACKBONE_ROOT / "EEGPT",
        "clone": True,
        "required": ["downstream", "pretrain"],
        "weight": "",
    },
    {
        "model_name": "MFrFM",
        "priority": "optional_frequency_domain",
        "repo_url": "https://github.com/zshubin/MFrFM-for-cross-task-EEG-pre-training.git",
        "local_path": BACKBONE_ROOT / "MFrFM",
        "clone": True,
        "required": [],
        "weight": "",
    },
    {
        "model_name": "EEG-DINO",
        "priority": "optional_hf_only_unresolved_github",
        "repo_url": "https://huggingface.co/eegdino/EEG-DINO",
        "local_path": BACKBONE_ROOT / "EEG-DINO",
        "clone": False,
        "required": [],
        "weight": "",
        "unresolved": True,
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_ROOT.mkdir(parents=True, exist_ok=True)
    log: List[str] = [
        "# Backbone Download Log",
        "",
        f"- workspace: `{WORKSPACE}`",
        f"- backbone_root: `{BACKBONE_ROOT}`",
        f"- training_run_started: `False`",
        "",
    ]
    rows = []
    for spec in BACKBONES:
        if spec["model_name"] == "CBraMod":
            spec["local_path"] = find_cbramod_path()
        rows.append(process_backbone(spec, log))
    write_csv(OUT_DIR / "backbone_inventory.csv", rows)
    (OUT_DIR / "backbone_inventory.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / "backbone_download_log.md").write_text("\n".join(log) + "\n")
    print(json.dumps({"status": "completed", "rows": rows}, indent=2, sort_keys=True))


def process_backbone(spec: Dict[str, object], log: List[str]) -> Dict[str, object]:
    name = str(spec["model_name"])
    local_path = Path(spec["local_path"])
    repo_url = str(spec["repo_url"])
    log.append(f"## {name}")
    if spec.get("unresolved"):
        log.append("- official GitHub not identified; HuggingFace model card found, no third-party clone attempted.")
        return make_row(spec, exists=False, git_commit="", weight_status="not_attempted_hf_only", notes="Official GitHub unresolved; HF page https://huggingface.co/eegdino/EEG-DINO found.", status="unresolved")

    if spec.get("clone"):
        if local_path.exists() and (local_path / ".git").exists():
            code, out = run(["git", "-C", str(local_path), "pull", "--ff-only"], timeout=120)
            log.append(f"- git pull: `{code}`\n\n```text\n{out[-3000:]}\n```")
        elif local_path.exists():
            log.append("- local path exists but is not a git repository; leaving untouched.")
        else:
            code, out = run(["git", "clone", repo_url, str(local_path)], timeout=240)
            log.append(f"- git clone: `{code}`\n\n```text\n{out[-3000:]}\n```")

    weight_status = "not_required_or_manual"
    if name == "MIRepNet":
        weight_status = try_download_mirepnet_weight(local_path, log)

    row = make_row(spec, exists=local_path.exists(), git_commit=git_commit(local_path), weight_status=weight_status)
    row["notes"] = notes_for(local_path, name, row)
    row["status"] = status_for(row)
    log.append(f"- status: `{row['status']}`")
    log.append("")
    return row


def try_download_mirepnet_weight(local_path: Path, log: List[str]) -> str:
    weight_path = local_path / "weight" / "MIRepNet.pth"
    if weight_path.exists():
        return "already_present"
    try:
        import huggingface_hub  # type: ignore
    except Exception as exc:
        log.append(f"- MIRepNet weight not downloaded: huggingface_hub unavailable: {exc}")
        return "missing_huggingface_hub"
    try:
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = huggingface_hub.hf_hub_download(
            repo_id="starself/MIRepNet",
            filename="MIRepNet.pth",
            local_dir=str(weight_path.parent),
            local_dir_use_symlinks=False,
        )
        log.append(f"- MIRepNet weight downloaded from HuggingFace: `{downloaded}`")
        return "downloaded_official_huggingface"
    except Exception as exc:
        log.append(f"- MIRepNet weight download failed: {exc}")
        return f"download_failed: {exc}"


def make_row(spec: Dict[str, object], exists: bool, git_commit: str, weight_status: str, notes: str = "", status: Optional[str] = None) -> Dict[str, object]:
    local_path = Path(spec["local_path"])
    readme = find_readme(local_path)
    req = local_path / "requirements.txt"
    weight_rel = str(spec.get("weight", ""))
    weight_path = local_path / weight_rel if weight_rel else None
    has_weight = bool(weight_path and weight_path.exists()) or has_any_checkpoint(local_path)
    row = {
        "model_name": spec["model_name"],
        "priority": spec["priority"],
        "repo_url": spec["repo_url"],
        "local_path": str(local_path),
        "exists": exists,
        "git_commit": git_commit,
        "has_requirements": req.exists(),
        "has_readme": bool(readme),
        "has_main_or_finetune_script": has_main_script(local_path),
        "has_pretrained_weight": has_weight,
        "weight_path": str(weight_path) if weight_path else checkpoint_paths(local_path),
        "weight_download_status": weight_status,
        "expected_input_hint": expected_input_hint(local_path, str(spec["model_name"])),
        "notes": notes,
        "status": status or "",
    }
    if not row["notes"]:
        row["notes"] = notes_for(local_path, str(spec["model_name"]), row)
    if not row["status"]:
        row["status"] = status_for(row)
    return row


def notes_for(local_path: Path, name: str, row: Dict[str, object]) -> str:
    if not local_path.exists():
        return "Repository/code path missing or clone failed."
    bits = []
    if name == "LaBraM":
        bits.append("Weight download is README/manual if no checkpoint exists.")
        bits.append("Check channel configuration before downstream use.")
    if name == "EEGPT":
        bits.append("Review README for sampling rate/channel/patch constraints before use.")
    if name == "MIRepNet" and not row.get("has_pretrained_weight"):
        bits.append("MIRepNet weight missing; install huggingface_hub or download official starself/MIRepNet weight manually.")
    readme = find_readme(local_path)
    if readme:
        text = safe_read(readme, 8000).lower()
        for token in ["download", "checkpoint", "pretrain", "channel", "sampling", "dataset"]:
            if token in text:
                bits.append(f"README mentions {token}.")
    return " ".join(dict.fromkeys(bits)) or "Code inspected."


def status_for(row: Dict[str, object]) -> str:
    if not row["exists"]:
        return "unresolved"
    if row["model_name"] == "CBraMod":
        return "ready" if row["has_pretrained_weight"] else "weight_missing"
    if row["model_name"] == "MIRepNet":
        return "ready" if row["has_pretrained_weight"] else "weight_missing"
    if row["has_pretrained_weight"]:
        return "ready"
    return "code_only"


def run(cmd: List[str], timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 999, repr(exc)


def git_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    code, out = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    return out.strip() if code == 0 else ""


def find_readme(path: Path) -> Optional[Path]:
    for name in ["README.md", "README.rst", "readme.md", "Readme.md"]:
        p = path / name
        if p.exists():
            return p
    return None


def safe_read(path: Path, n: int) -> str:
    try:
        return path.read_text(errors="ignore")[:n]
    except Exception:
        return ""


def has_main_script(path: Path) -> bool:
    names = ["finetune.py", "finetune_main.py", "run_class_finetuning.py", "quick_example.py", "main.py"]
    return any((path / name).exists() for name in names) or any(path.glob("**/*finetun*.py")) or any(path.glob("**/*linear*probe*.py"))


def has_any_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    patterns = ["*.pth", "*.pt", "*.ckpt", "*.bin", "*.safetensors"]
    return any(any(path.glob(f"**/{pat}")) for pat in patterns)


def checkpoint_paths(path: Path) -> str:
    if not path.exists():
        return ""
    found = []
    for pat in ["*.pth", "*.pt", "*.ckpt", "*.bin", "*.safetensors"]:
        found.extend(str(p) for p in path.glob(f"**/{pat}"))
    return ";".join(found[:8])


def expected_input_hint(path: Path, name: str) -> str:
    readme = find_readme(path)
    text = safe_read(readme, 15000) if readme else ""
    if name == "CBraMod":
        return "[B,C,S,200], e.g. [B,22,4,200] for BCIC2a"
    hints = []
    for key in ["sampling", "channel", "patch", "input", "dataset"]:
        idx = text.lower().find(key)
        if idx >= 0:
            hints.append(text[idx : idx + 220].replace("\n", " "))
    return " | ".join(hints[:3])


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_cbramod_path() -> Path:
    candidates = [
        WORKSPACE / "CBraMod-main",
        WORKSPACE / "sas_cert_cbramod_mve" / "third_party" / "CBraMod",
        WORKSPACE.parent / "新研究" / "CBraMod-main",
        WORKSPACE.parent / "CBraMod-main",
    ]
    existing = [p for p in candidates if (p / "models" / "cbramod.py").exists()]
    with_weight = [p for p in existing if (p / "pretrained_weights" / "pretrained_weights.pth").exists()]
    return with_weight[0] if with_weight else (existing[0] if existing else candidates[0])


if __name__ == "__main__":
    main()
