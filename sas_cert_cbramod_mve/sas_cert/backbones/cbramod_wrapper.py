from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class CBraModFeatureExtractor:
    def __init__(self, cbramod_root: str, device: str = "auto"):
        self.cbramod_root = Path(cbramod_root).resolve()
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        if str(self.cbramod_root) not in sys.path:
            sys.path.insert(0, str(self.cbramod_root))
        from models.cbramod import CBraMod

        self.model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        self.checkpoint_audit = self._load_weights()
        self.model.proj_out = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_weights(self) -> Dict[str, object]:
        weight_path = self.cbramod_root / "pretrained_weights" / "pretrained_weights.pth"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing CBraMod weights: {weight_path}")
        ckpt = torch.load(weight_path, map_location="cpu")
        model_state = self.model.state_dict()
        matched = []
        shape_mismatch = []
        missing = []
        for key, value in model_state.items():
            if key in ckpt:
                if tuple(value.shape) == tuple(ckpt[key].shape):
                    matched.append(key)
                else:
                    shape_mismatch.append({"key": key, "model": list(value.shape), "checkpoint": list(ckpt[key].shape)})
            else:
                missing.append(key)
        extra = [key for key in ckpt.keys() if key not in model_state]
        if shape_mismatch or extra:
            raise RuntimeError(f"CBraMod checkpoint blocker: shape_mismatch={len(shape_mismatch)} extra={len(extra)}")
        self.model.load_state_dict(ckpt, strict=False)
        core_missing = [key for key in missing if "adapter" not in key and "proj_out" not in key and "head" not in key]
        if core_missing:
            raise RuntimeError(f"CBraMod core missing keys detected: {core_missing[:20]}")
        return {
            "checkpoint_path": str(weight_path),
            "checkpoint_total_keys": len(ckpt),
            "model_total_keys": len(model_state),
            "matched_keys": len(matched),
            "missing_keys": len(missing),
            "unexpected_keys": len(extra),
            "shape_mismatch": shape_mismatch,
            "missing_keys_preview": missing[:30],
            "core_missing_keys": core_missing,
            "adapter_missing_accepted": len(missing) > 0 and len(core_missing) == 0,
        }

    @staticmethod
    def to_cbramod_input(x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[1:] != (22, 800):
            raise ValueError(f"Expected [N,22,800], got {x.shape}")
        return x.reshape(x.shape[0], 22, 4, 200).astype(np.float32)

    def extract(self, x: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        x_patch = self.to_cbramod_input(x)
        reps = []
        pooled = []
        with torch.no_grad():
            for start in range(0, len(x_patch), batch_size):
                batch = torch.from_numpy(x_patch[start : start + batch_size]).float().to(self.device)
                out = self.model(batch)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError("CBraMod feature contains NaN/Inf.")
                reps.append(out.detach().cpu().numpy().astype(np.float32))
                pooled.append(out.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(pooled, axis=0), np.concatenate(reps, axis=0)

