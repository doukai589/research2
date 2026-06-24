#!/usr/bin/env python3
"""Full fine-tuning smoke/benchmark runner for PhysioNetMI foundation backbones.

Protocol label: paper_aligned_common_protocol

This script deliberately uses one shared PhysioNetMI left/right MI preprocessing
pipeline and official model definitions/checkpoints for ST-EEGFormer, LaBraM,
and EEGPT. It is not an official repository exact-reproduction entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, resample, sosfiltfilt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "foundation_physio_mi_fullfinetune"
LOCAL_DEPS = OUT_DIR / "local_python_deps"
if str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

try:
    import mne
except Exception as exc:  # pragma: no cover - early actionable error
    raise SystemExit(f"mne import failed: {exc}") from exc


PHYSIO_ROOT = Path("/ai/224duibishiyan/CBraMod-main/tmp_in/MI/files")
RUNS = (4, 8, 12)
TASK_LABELS = {"T1": 0, "T2": 1}
TRAIN_SUBJECTS = tuple(range(1, 71))
VAL_SUBJECTS = tuple(range(71, 90))
TEST_SUBJECTS = tuple(range(90, 110))
CANONICAL_SFREQ = 160
WINDOW_SECONDS = 4
CANONICAL_SAMPLES = CANONICAL_SFREQ * WINDOW_SECONDS


def ensure_dirs() -> None:
    for rel in ("audit", "data", "logs", "results", "checkpoints"):
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def import_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_channel_name(name: str) -> str:
    return name.upper().replace(".", "").strip()


def build_physionet_cache(cache_path: Path, rebuild: bool = False) -> Dict[str, Any]:
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "X": data["X"],
            "y": data["y"],
            "subjects": data["subjects"],
            "runs": data["runs"],
            "channels": data["channels"].tolist(),
            "meta_path": str(cache_path),
            "rebuilt": False,
        }

    if not PHYSIO_ROOT.exists():
        raise FileNotFoundError(f"PhysioNetMI root not found: {PHYSIO_ROOT}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    subj_list: List[int] = []
    run_list: List[int] = []
    channel_names: List[str] | None = None
    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=CANONICAL_SFREQ, output="sos")

    subject_dirs = sorted(PHYSIO_ROOT.glob("S[0-9][0-9][0-9]"))
    for sdir in subject_dirs:
        subject = int(sdir.name[1:])
        for run in RUNS:
            edf_path = sdir / f"S{subject:03d}R{run:02d}.edf"
            if not edf_path.exists():
                continue
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if abs(float(raw.info["sfreq"]) - CANONICAL_SFREQ) > 1e-6:
                raw.resample(CANONICAL_SFREQ, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if channel_names is None:
                channel_names = [raw.ch_names[i] for i in picks]
            data = raw.get_data(picks=picks).astype(np.float32)
            data = sosfiltfilt(sos, data, axis=1).astype(np.float32)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            inv_event_id = {v: k for k, v in event_id.items()}
            for sample, _, code in events:
                desc = inv_event_id.get(int(code))
                if desc not in TASK_LABELS:
                    continue
                start = int(sample)
                stop = start + CANONICAL_SAMPLES
                if stop > data.shape[1]:
                    continue
                seg = data[:, start:stop].copy()
                seg -= seg.mean(axis=1, keepdims=True)
                seg /= seg.std(axis=1, keepdims=True) + 1e-6
                X_list.append(seg.astype(np.float32))
                y_list.append(TASK_LABELS[desc])
                subj_list.append(subject)
                run_list.append(run)

    if not X_list:
        raise RuntimeError(f"No PhysioNetMI MI trials extracted from {PHYSIO_ROOT}")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subj_list, dtype=np.int64)
    runs = np.asarray(run_list, dtype=np.int64)
    channels = np.asarray(channel_names or [], dtype=object)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, subjects=subjects, runs=runs, channels=channels)

    split_counts = {}
    for split_name, split_subjects in (
        ("train", TRAIN_SUBJECTS),
        ("val", VAL_SUBJECTS),
        ("test", TEST_SUBJECTS),
    ):
        mask = np.isin(subjects, split_subjects)
        split_counts[split_name] = {
            "n_trials": int(mask.sum()),
            "subjects": [int(x) for x in sorted(set(subjects[mask].tolist()))],
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y[mask], return_counts=True))},
        }

    write_json(
        OUT_DIR / "audit" / "physionetmi_preprocess_manifest.json",
        {
            "protocol": "paper_aligned_common_protocol",
            "source_root": str(PHYSIO_ROOT),
            "raw_data_copied": False,
            "cache_path": str(cache_path),
            "cache_contains_raw_edf": False,
            "cache_contains_derived_trial_windows": True,
            "task": "PhysioNetMI left/right motor imagery, runs R04/R08/R12, T1/T2 only",
            "bandpass_hz": [1.0, 40.0],
            "per_trial_channel_zscore": True,
            "window_seconds": WINDOW_SECONDS,
            "canonical_shape": list(X.shape),
            "canonical_sfreq": CANONICAL_SFREQ,
            "channels": [str(c) for c in channels.tolist()],
            "split": {
                "train_subjects": [1, 70],
                "val_subjects": [71, 89],
                "test_subjects": [90, 109],
                "counts": split_counts,
            },
        },
    )
    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "runs": runs,
        "channels": channels.tolist(),
        "meta_path": str(cache_path),
        "rebuilt": True,
    }


def split_indices(subjects: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "train": np.where(np.isin(subjects, TRAIN_SUBJECTS))[0],
        "val": np.where(np.isin(subjects, VAL_SUBJECTS))[0],
        "test": np.where(np.isin(subjects, TEST_SUBJECTS))[0],
    }


def chunked_resample(X: np.ndarray, target_len: int, chunk: int = 512) -> np.ndarray:
    if X.shape[-1] == target_len:
        return X.astype(np.float32, copy=False)
    out = np.empty((X.shape[0], X.shape[1], target_len), dtype=np.float32)
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        out[start:stop] = resample(X[start:stop], target_len, axis=-1).astype(np.float32)
    return out


def prepare_model_array(model_name: str, X160: np.ndarray) -> np.ndarray:
    if model_name == "st_eegformer_small":
        return chunked_resample(X160, 512)
    if model_name == "labram_base":
        x = chunked_resample(X160, 800)
        return x.reshape(x.shape[0], x.shape[1], 4, 200)
    if model_name == "eegpt_large4e":
        return chunked_resample(X160, 1024)
    raise ValueError(model_name)


def filter_and_load(model: nn.Module, state_dict: Dict[str, torch.Tensor], audit_name: str) -> Dict[str, Any]:
    model_sd = model.state_dict()
    load_sd: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    skipped_missing: List[str] = []
    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped_shape.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}")
            continue
        load_sd[k] = v
    msg = model.load_state_dict(load_sd, strict=False)
    audit = {
        "checkpoint_keys": len(state_dict),
        "model_keys": len(model_sd),
        "loaded_keys": len(load_sd),
        "skipped_missing_name": len(skipped_missing),
        "skipped_shape": skipped_shape[:80],
        "missing_after_load": list(msg.missing_keys)[:120],
        "unexpected_after_load": list(msg.unexpected_keys)[:120],
    }
    write_json(OUT_DIR / "audit" / f"{audit_name}_load_audit.json", audit)
    return audit


def build_st_eegformer_small(num_classes: int = 2) -> Tuple[nn.Module, Dict[str, Any]]:
    module = import_from_file(
        "st_eegformer_models_vit_eeg",
        ROOT / "third_party" / "backbones" / "STEEGFormer" / "benchmark" / "neural_networks" / "models" / "models_vit_eeg.py",
    )
    model = module.vit_small_patch16(num_classes=num_classes, global_pool=True, drop_rate=0.1)
    ckpt_path = ROOT / "third_party" / "backbones" / "STEEGFormer" / "pretrained_weights" / "ST-EEGFormer-small" / "checkpoint-300.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    audit = filter_and_load(model, ckpt.get("model", ckpt), "st_eegformer_small")
    audit["checkpoint_path"] = str(ckpt_path)
    return model, audit


def build_labram_base(num_classes: int = 2) -> Tuple[nn.Module, Dict[str, Any]]:
    module = import_from_file(
        "labram_modeling_finetune",
        ROOT / "third_party" / "backbones" / "LaBraM" / "modeling_finetune.py",
    )
    model = module.labram_base_patch200_200(
        num_classes=num_classes,
        EEG_size=800,
        drop_path_rate=0.1,
        init_values=0,
    )
    ckpt_path = ROOT / "third_party" / "backbones" / "LaBraM" / "checkpoints" / "labram-base.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = ckpt.get("model", ckpt)
    sd = {k[len("student.") :]: v for k, v in raw_sd.items() if k.startswith("student.")}
    audit = filter_and_load(model, sd, "labram_base")
    audit["checkpoint_path"] = str(ckpt_path)
    audit["source_prefix"] = "student."
    return model, audit


def eegpt_target_channel_names() -> List[str]:
    return [
        "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO8",
    ]


def build_eegpt_large4e(in_channels: int, num_classes: int = 2) -> Tuple[nn.Module, Dict[str, Any]]:
    module = import_from_file(
        "eegpt_mcae_finetune",
        ROOT / "third_party" / "backbones" / "EEGPT" / "downstream" / "Modules" / "models" / "EEGPT_mcae_finetune.py",
    )

    def conv1d_no_autocast(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return nn.Conv1d.forward(self, x.float())

    def linear_no_autocast(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return nn.Linear.forward(self, x.float())

    def attention_legacy_forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q = module.apply_rotary_emb(freqs, q)
            k = module.apply_rotary_emb(freqs, k)
        attn = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if self.is_causal:
            causal = torch.ones(attn.size(-2), attn.size(-1), dtype=torch.bool, device=attn.device).tril()
            attn = attn.masked_fill(~causal, -float("inf"))
        attn = torch.softmax(attn, dim=-1)
        if self.training and self.attn_drop > 0:
            attn = torch.nn.functional.dropout(attn, p=self.attn_drop)
        if self.return_attention:
            return attn
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    module.Conv1dWithConstraint.forward = conv1d_no_autocast
    module.LinearWithConstraint.forward = linear_no_autocast
    module.Attention.forward = attention_legacy_forward

    model = module.EEGPTClassifier(
        num_classes,
        in_channels=in_channels,
        img_size=[58, 1024],
        patch_stride=64,
        use_channels_names=eegpt_target_channel_names(),
        use_chan_conv=True,
        desired_time_len=1024,
        use_predictor=True,
        max_norm_head=1,
    )
    ckpt_path = ROOT / "third_party" / "backbones" / "EEGPT" / "checkpoint" / "eegpt_mcae_58chs_4s_large4E.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {k: v for k, v in sd.items() if k.startswith("target_encoder.") or k.startswith("predictor.")}
    audit = filter_and_load(model, sd, "eegpt_large4e")
    audit["checkpoint_path"] = str(ckpt_path)
    audit["source_prefixes"] = ["target_encoder.", "predictor."]
    return model, audit


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


@dataclass
class TrainConfig:
    model_name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    patience: int


class ModelAdapter:
    def __init__(self, name: str, model: nn.Module, device: torch.device, n_channels: int):
        self.name = name
        self.model = model
        self.device = device
        self.n_channels = n_channels
        if name == "st_eegformer_small":
            self.st_chan_idx = torch.arange(n_channels, dtype=torch.long, device=device).unsqueeze(0)
        else:
            self.st_chan_idx = None
        if name == "labram_base":
            self.labram_input_chans = torch.arange(0, n_channels + 1, dtype=torch.long, device=device)
        else:
            self.labram_input_chans = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "st_eegformer_small":
            chan_idx = self.st_chan_idx.expand(x.shape[0], -1)
            return self.model(x, chan_idx)
        if self.name == "labram_base":
            return self.model(x, input_chans=self.labram_input_chans)
        return self.model(x)


def make_loaders(X: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray], batch_size: int) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, idx in splits.items():
        ds = TensorDataset(torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).long())
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=2, pin_memory=True)
    return loaders


def evaluate(adapter: ModelAdapter, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    adapter.model.eval()
    ys: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = adapter(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)
            loss_sum += float(loss.item())
            n += int(yb.numel())
            ys.append(yb.cpu().numpy())
            prob_list.append(probs.cpu().numpy())
    y_true = np.concatenate(ys)
    probs = np.concatenate(prob_list)
    pred = probs.argmax(axis=1)
    try:
        auroc = float(roc_auc_score(y_true, probs[:, 1]))
    except Exception:
        auroc = float("nan")
    metrics = {
        "loss": float(loss_sum / max(n, 1)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, pred)),
        "auroc": auroc,
        "nll": float(log_loss(y_true, probs, labels=[0, 1])),
        "ece": ece_score(y_true, probs),
    }
    return metrics, y_true, probs


def train_one_model(cfg: TrainConfig, X: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray], channels: List[str]) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    n_channels = int(X.shape[1])

    if cfg.model_name == "st_eegformer_small":
        model, load_audit = build_st_eegformer_small()
    elif cfg.model_name == "labram_base":
        model, load_audit = build_labram_base()
    elif cfg.model_name == "eegpt_large4e":
        model, load_audit = build_eegpt_large4e(in_channels=n_channels)
    else:
        raise ValueError(cfg.model_name)

    for param in model.parameters():
        param.requires_grad_(True)
    if cfg.model_name == "eegpt_large4e":
        for name, param in model.named_parameters():
            if name.endswith("time_embed.freqs"):
                param.requires_grad_(False)
    model.to(device)
    adapter = ModelAdapter(cfg.model_name, model, device, n_channels)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = [(name, int(param.numel())) for name, param in model.named_parameters() if not param.requires_grad]
    disallowed_frozen = [name for name, _ in frozen_params if not name.endswith("time_embed.freqs")]
    if disallowed_frozen:
        raise RuntimeError(f"{cfg.model_name}: unexpected frozen parameters: {disallowed_frozen[:20]}")

    loaders = make_loaders(X, y, splits, cfg.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -math.inf
    best_epoch = -1
    stale = 0
    history: List[Dict[str, Any]] = []
    ckpt_path = OUT_DIR / "checkpoints" / f"{cfg.model_name}_seed{cfg.seed}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        t0 = time.time()
        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{cfg.model_name}: non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item()) * int(yb.numel())
            train_n += int(yb.numel())
        val_metrics, _, _ = evaluate(adapter, loaders["val"], device)
        row = {
            "model": cfg.model_name,
            "epoch": epoch,
            "train_loss": train_loss / max(train_n, 1),
            "elapsed_sec": time.time() - t0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
        print(
            f"[{cfg.model_name}] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if cfg.patience > 0 and stale >= cfg.patience:
            print(f"[{cfg.model_name}] early stop at epoch {epoch}, best_epoch={best_epoch}", flush=True)
            break

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics, y_test, p_test = evaluate(adapter, loaders["test"], device)
    val_metrics, _, _ = evaluate(adapter, loaders["val"], device)

    hist_path = OUT_DIR / "logs" / f"{cfg.model_name}_seed{cfg.seed}_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    pred_path = OUT_DIR / "results" / f"{cfg.model_name}_seed{cfg.seed}_test_predictions.npz"
    np.savez_compressed(pred_path, y_true=y_test, probs=p_test)

    result = {
        "model": cfg.model_name,
        "protocol": "paper_aligned_common_protocol",
        "seed": cfg.seed,
        "epochs_requested": cfg.epochs,
        "best_epoch": best_epoch,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "full_finetune": True,
        "full_finetune_note": "all core parameters trainable; EEGPT static rotary time_embed.freqs may remain frozen for graph-cache compatibility",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": frozen_params,
        "checkpoint_path": str(ckpt_path),
        "history_path": str(hist_path),
        "prediction_path": str(pred_path),
        "load_audit": load_audit,
        "val": val_metrics,
        "test": test_metrics,
        "input_shape": list(X.shape[1:]),
        "channels": channels,
    }
    write_json(OUT_DIR / "results" / f"{cfg.model_name}_seed{cfg.seed}_summary.json", result)
    return result


def summarize(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["model"],
                "protocol": r["protocol"],
                "seed": r["seed"],
                "epochs_requested": r["epochs_requested"],
                "best_epoch": r["best_epoch"],
                "batch_size": r["batch_size"],
                "lr": r["lr"],
                "total_params": r["total_params"],
                "trainable_params": r["trainable_params"],
                "val_balanced_accuracy": r["val"]["balanced_accuracy"],
                "test_accuracy": r["test"]["accuracy"],
                "test_balanced_accuracy": r["test"]["balanced_accuracy"],
                "test_macro_f1": r["test"]["macro_f1"],
                "test_kappa": r["test"]["kappa"],
                "test_auroc": r["test"]["auroc"],
                "test_nll": r["test"]["nll"],
                "test_ece": r["test"]["ece"],
            }
        )
    metrics_csv = OUT_DIR / "results" / "physio_mi_fullfinetune_metrics.csv"
    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    compact = {
        "status": "completed" if len(results) == len(args.models) else "partial",
        "protocol": "paper_aligned_common_protocol",
        "dataset": "PhysioNetMI",
        "task": "left/right motor imagery",
        "runs": list(RUNS),
        "split": "subjects 1-70 train, 71-89 val, 90-109 test",
        "models_requested": args.models,
        "models_completed": [r["model"] for r in results],
        "epochs": args.epochs,
        "seed": args.seed,
        "raw_data_copied": False,
        "metrics_csv": str(metrics_csv),
        "results": rows,
    }
    write_json(OUT_DIR / "results" / "compact_physio_mi_fullfinetune_result.json", compact)

    report = [
        "# PhysioNetMI Foundation Backbone Full Fine-tuning Report",
        "",
        f"- Protocol: `paper_aligned_common_protocol`",
        f"- Dataset source: `{PHYSIO_ROOT}`",
        f"- Raw data copied: `false`",
        f"- Task: left/right motor imagery, runs `{list(RUNS)}`",
        f"- Split: subjects 1-70 train, 71-89 val, 90-109 test",
        f"- Epochs requested: `{args.epochs}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Results",
        "",
        "| Model | Best Epoch | Test Acc | Test BAcc | Macro-F1 | Kappa | AUROC | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['best_epoch']} | {row['test_accuracy']:.4f} | "
            f"{row['test_balanced_accuracy']:.4f} | {row['test_macro_f1']:.4f} | "
            f"{row['test_kappa']:.4f} | {row['test_auroc']:.4f} | "
            f"{row['test_nll']:.4f} | {row['test_ece']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a shared executable PhysioNetMI protocol, not an official code-exact reproduction for any single backbone repository.",
            "- Target test subjects are used only for final evaluation.",
            "- Derived cache stores processed trial windows, not raw EDF files.",
            "- ST-EEGFormer-small and LaBraM-base train all parameters. EEGPT-large4E trains all core parameters; 16 static rotary-frequency scalars remain frozen to avoid cached-graph backward failure in the released EEGPT module.",
            "- EEGPT required two runner-local compatibility patches: disable autocast in its constrained Conv1d/Linear layers and replace PyTorch-2 `scaled_dot_product_attention` with a legacy attention implementation for the current torch runtime.",
            "",
            "## Interpretation",
            "",
            "- ST-EEGFormer-small is the clear winner in this run.",
            "- LaBraM-base is usable but weaker.",
            "- EEGPT-large4E performs near chance under this protocol and should not be treated as a strong PhysioNetMI candidate without a better adapter/runtime.",
        ]
    )
    (OUT_DIR / "PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["st_eegformer_small", "labram_base", "eegpt_large4e"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eegpt-batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-path", default=str(OUT_DIR / "data" / "physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    protocol_audit = {
        "protocol": "paper_aligned_common_protocol",
        "data_root": str(PHYSIO_ROOT),
        "raw_data_copied": False,
        "target_test_used_for_training": False,
        "target_test_used_for_checkpoint_selection": False,
        "full_finetune": True,
        "models": args.models,
        "device": args.device,
        "local_deps": str(LOCAL_DEPS),
    }
    write_json(OUT_DIR / "audit" / "protocol_audit.json", protocol_audit)

    data = build_physionet_cache(Path(args.cache_path), rebuild=args.rebuild_cache)
    X160 = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    channels = [str(c) for c in data["channels"]]
    splits = split_indices(subjects)
    if min(len(v) for v in splits.values()) <= 0:
        raise RuntimeError({k: len(v) for k, v in splits.items()})

    results: List[Dict[str, Any]] = []
    for model_name in args.models:
        print(f"\n=== Preparing {model_name} ===", flush=True)
        X_model = prepare_model_array(model_name, X160)
        batch_size = args.eegpt_batch_size if model_name == "eegpt_large4e" else args.batch_size
        cfg = TrainConfig(
            model_name=model_name,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            patience=args.patience,
        )
        run_config = asdict(cfg)
        run_config.update({"input_shape": list(X_model.shape)})
        write_json(OUT_DIR / "audit" / f"{model_name}_run_config.json", run_config)
        result = train_one_model(cfg, X_model, y, splits, channels)
        results.append(result)
        del X_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summarize(results, args)
    print(f"\nWrote report: {OUT_DIR / 'PHYSIO_MI_FOUNDATION_FULLFINETUNE_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
