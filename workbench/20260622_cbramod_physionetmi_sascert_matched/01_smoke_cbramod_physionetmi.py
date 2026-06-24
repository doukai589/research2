#!/usr/bin/env python3
"""Smoke test CBraMod frozen features on the shared PhysioNetMI cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sas_core.backbones.cbramod import build_cbramod
from sas_core.data.physionet_mi import build_physionet_mi_cache, default_physionet_mi_protocol, split_indices
from sas_core.utils.io import write_json
from sas_core.utils.seed import set_seed


OUT_DIR = Path(__file__).resolve().parent / "outputs"
CANONICAL_CACHE = ROOT / "outputs" / "foundation_physio_mi_fullfinetune" / "data" / "physionetmi_lr_r04_r08_r12_160hz_4s_zscore.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(3407)

    protocol = default_physionet_mi_protocol()
    data = build_physionet_mi_cache(CANONICAL_CACHE, protocol=protocol, rebuild=False)
    splits = split_indices(data["subjects"], protocol)
    sample_idx = splits["train"][: args.n_samples]
    x = data["X"][sample_idx].astype(np.float32)
    y = data["y"][sample_idx]
    subjects = data["subjects"][sample_idx]

    model, audit = build_cbramod(n_channels=x.shape[1])
    model.to(args.device)
    patch = model.to_cbramod_input(x)
    pooled, reps = model.extract_features(x, args.device, batch_size=args.n_samples)

    report = {
        "status": "passed",
        "dataset": "PhysioNetMI",
        "raw_data_copied": False,
        "cache_path": str(CANONICAL_CACHE),
        "cache_shape": list(data["X"].shape),
        "sample_indices": [int(i) for i in sample_idx.tolist()],
        "sample_subjects": [int(i) for i in subjects.tolist()],
        "sample_labels": [int(i) for i in y.tolist()],
        "raw_shape": list(x.shape),
        "cbramod_input_shape": list(patch.shape),
        "representation_shape": list(reps.shape),
        "pooled_feature_shape": list(pooled.shape),
        "pooled_feature_dim": int(pooled.shape[1]),
        "feature_nan_count": int(np.isnan(pooled).sum() + np.isnan(reps).sum()),
        "feature_inf_count": int(np.isinf(pooled).sum() + np.isinf(reps).sum()),
        "feature_mean": float(pooled.mean()),
        "feature_std": float(pooled.std()),
        "device": str(args.device),
        "split_counts": {name: int(len(idx)) for name, idx in splits.items()},
        "checkpoint_audit": audit,
        "protocol_checks": {
            "physionet_loader_reused": True,
            "target_samples_resampled_to_800_for_cbramod": True,
            "input_is_64_channels_4_patches_200_samples": list(patch.shape[1:]) == [64, 4, 200],
            "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
            "no_nan_inf": int(np.isnan(pooled).sum() + np.isnan(reps).sum() + np.isinf(pooled).sum() + np.isinf(reps).sum()) == 0,
        },
    }
    if not report["protocol_checks"]["input_is_64_channels_4_patches_200_samples"]:
        report["status"] = "failed"
    if not report["protocol_checks"]["no_nan_inf"]:
        report["status"] = "failed"
    write_json(OUT_DIR / "cbramod_smoke_report.json", report)
    (OUT_DIR / "cbramod_smoke_log.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
