"""ST-EEGFormer wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from sas_core.backbones.import_utils import import_from_file


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "local_python_deps"


@dataclass(frozen=True)
class STEEGFormerConfig:
    variant: str = "small"
    num_classes: int = 2
    global_pool: bool = True
    drop_rate: float = 0.1
    code_path: Path = ROOT / "third_party" / "backbones" / "STEEGFormer"
    checkpoint_path: Path = (
        ROOT
        / "third_party"
        / "backbones"
        / "STEEGFormer"
        / "pretrained_weights"
        / "ST-EEGFormer-small"
        / "checkpoint-300.pth"
    )


class STEEGFormerClassifier(nn.Module):
    """Thin callable wrapper that supplies channel indices to ST-EEGFormer."""

    def __init__(self, model: nn.Module, n_channels: int):
        super().__init__()
        self.model = model
        self.n_channels = n_channels
        self.register_buffer("channel_indices", torch.arange(n_channels, dtype=torch.long).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_indices = self.channel_indices.expand(x.shape[0], -1)
        return self.model(x, channel_indices)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        channel_indices = self.channel_indices.expand(x.shape[0], -1)
        return self.model.forward_features(x, channel_indices)


def _ensure_local_deps() -> None:
    import sys

    if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
        sys.path.insert(0, str(LOCAL_DEPS))


def build_steegformer(config: STEEGFormerConfig | None = None, n_channels: int = 64) -> tuple[STEEGFormerClassifier, Dict[str, Any]]:
    config = config or STEEGFormerConfig()
    _ensure_local_deps()
    module = import_from_file(
        f"steegformer_models_vit_eeg_{config.variant}",
        config.code_path / "benchmark" / "neural_networks" / "models" / "models_vit_eeg.py",
    )
    if config.variant == "small":
        model = module.vit_small_patch16(
            num_classes=config.num_classes,
            global_pool=config.global_pool,
            drop_rate=config.drop_rate,
        )
    elif config.variant == "base":
        model = module.vit_base_patch16(
            num_classes=config.num_classes,
            global_pool=config.global_pool,
            drop_rate=config.drop_rate,
        )
    else:
        raise ValueError(f"unsupported ST-EEGFormer variant: {config.variant}")

    audit = load_steegformer_checkpoint(model, config.checkpoint_path)
    return STEEGFormerClassifier(model, n_channels=n_channels), audit


def load_steegformer_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}")
            continue
        load_state[key] = value
    msg = model.load_state_dict(load_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": len(state_dict),
        "model_keys": len(model_state),
        "loaded_keys": len(load_state),
        "skipped_missing_name": len(skipped_missing),
        "skipped_shape": skipped_shape[:80],
        "missing_after_load": list(msg.missing_keys)[:120],
        "unexpected_after_load": list(msg.unexpected_keys)[:120],
    }
