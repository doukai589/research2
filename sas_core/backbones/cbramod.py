"""CBraMod frozen feature wrapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from sas_core.data.transforms import chunked_resample


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CBraModConfig:
    code_path_candidates: tuple[Path, ...] = (
        ROOT / "third_party" / "CBraMod-main",
        ROOT.parent / "新研究" / "CBraMod-main",
    )
    checkpoint_candidates: tuple[Path, ...] = (
        ROOT / "third_party" / "CBraMod-main" / "pretrained_weights" / "pretrained_weights.pth",
        ROOT.parent / "新研究" / "CBraMod-main" / "pretrained_weights" / "pretrained_weights.pth",
    )
    patch_size: int = 200
    target_samples: int = 800


class CBraModFeatureExtractor(nn.Module):
    """Thin wrapper that exposes pooled CBraMod patch features."""

    def __init__(self, model: nn.Module, patch_size: int = 200, target_samples: int = 800):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.target_samples = target_samples

    def to_cbramod_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(f"expected [N,C,T], got {x.shape}")
        x800 = chunked_resample(x.astype(np.float32), self.target_samples)
        if x800.shape[-1] % self.patch_size != 0:
            raise ValueError(f"target samples {x800.shape[-1]} not divisible by patch size {self.patch_size}")
        patch_num = x800.shape[-1] // self.patch_size
        return x800.reshape(x800.shape[0], x800.shape[1], patch_num, self.patch_size).astype(np.float32)

    def forward(self, x_patch: torch.Tensor) -> torch.Tensor:
        return self.model(x_patch)

    def extract_features(self, x: np.ndarray, device: str, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        x_patch = self.to_cbramod_input(x)
        pooled: list[np.ndarray] = []
        reps: list[np.ndarray] = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(x_patch), batch_size):
                batch = torch.from_numpy(x_patch[start : start + batch_size]).float().to(device)
                out = self.model(batch)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError("CBraMod output contains NaN/Inf")
                reps.append(out.detach().cpu().numpy().astype(np.float32))
                pooled.append(out.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(pooled, axis=0), np.concatenate(reps, axis=0)


def _first_existing(paths: tuple[Path, ...], what: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"could not find {what}: {[str(p) for p in paths]}")


def build_cbramod(config: CBraModConfig | None = None, n_channels: int = 64) -> tuple[CBraModFeatureExtractor, Dict[str, Any]]:
    config = config or CBraModConfig()
    code_path = _first_existing(config.code_path_candidates, "CBraMod code path")
    checkpoint_path = _first_existing(config.checkpoint_candidates, "CBraMod checkpoint")
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    from models.cbramod import CBraMod

    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    audit = load_cbramod_checkpoint(model, checkpoint_path)
    model.proj_out = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    wrapper = CBraModFeatureExtractor(model, patch_size=config.patch_size, target_samples=config.target_samples)
    audit.update(
        {
            "code_path": str(code_path),
            "n_channels": n_channels,
            "target_samples": config.target_samples,
            "patch_size": config.patch_size,
            "patch_num": config.target_samples // config.patch_size,
            "proj_out_replaced_with_identity": True,
            "frozen_backbone": True,
        }
    )
    return wrapper, audit


def load_cbramod_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    missing_in_checkpoint: list[str] = []
    shape_mismatch: list[str] = []
    for key, value in model_state.items():
        if key not in state_dict:
            missing_in_checkpoint.append(key)
            continue
        if tuple(value.shape) != tuple(state_dict[key].shape):
            shape_mismatch.append(f"{key}: ckpt={tuple(state_dict[key].shape)} model={tuple(value.shape)}")
            continue
        load_state[key] = state_dict[key]
    unexpected = [key for key in state_dict.keys() if key not in model_state]
    if shape_mismatch or unexpected:
        raise RuntimeError(f"CBraMod checkpoint blocker: shape_mismatch={len(shape_mismatch)} unexpected={len(unexpected)}")
    msg = model.load_state_dict(load_state, strict=False)
    core_missing = [
        key
        for key in missing_in_checkpoint
        if "adapter" not in key and "proj_out" not in key and "head" not in key and "classifier" not in key
    ]
    if core_missing:
        raise RuntimeError(f"CBraMod core missing keys detected: {core_missing[:20]}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": len(state_dict),
        "model_keys": len(model_state),
        "loaded_keys": len(load_state),
        "missing_in_checkpoint": missing_in_checkpoint[:120],
        "missing_after_load": list(msg.missing_keys)[:120],
        "unexpected_after_load": list(msg.unexpected_keys)[:120],
        "shape_mismatch": shape_mismatch[:80],
        "unexpected_checkpoint_keys": unexpected[:80],
    }
