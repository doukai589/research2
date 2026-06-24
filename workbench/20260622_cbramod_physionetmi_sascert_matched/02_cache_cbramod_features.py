#!/usr/bin/env python3
"""Cache frozen CBraMod pooled features for the shared PhysioNetMI cache."""

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def extract_pooled_only(model, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    patch = model.to_cbramod_input(X)
    pooled: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patch), batch_size):
            stop = min(start + batch_size, len(patch))
            print(f"extracting {start}:{stop}", flush=True)
            batch = torch.from_numpy(patch[start:stop]).float().to(device)
            out = model(batch)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError(f"CBraMod output contains NaN/Inf in batch {start}:{stop}")
            pooled.append(out.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(pooled, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = OUT_DIR / "cbramod_original_features.npz"
    manifest_path = OUT_DIR / "cbramod_original_features_manifest.json"
    if feature_path.exists() and manifest_path.exists() and not args.rebuild:
        print(f"cache exists: {feature_path}", flush=True)
        return

    set_seed(3407)
    protocol = default_physionet_mi_protocol()
    data = build_physionet_mi_cache(CANONICAL_CACHE, protocol=protocol, rebuild=False)
    splits = split_indices(data["subjects"], protocol)
    model, audit = build_cbramod(n_channels=data["X"].shape[1])
    model.to(args.device)
    features = extract_pooled_only(model, data["X"].astype(np.float32), args.device, args.batch_size)
    if features.shape[0] != data["X"].shape[0]:
        raise RuntimeError(f"feature count mismatch: {features.shape[0]} vs {data['X'].shape[0]}")
    if np.isnan(features).any() or np.isinf(features).any():
        raise RuntimeError("feature cache contains NaN/Inf")

    np.savez_compressed(
        feature_path,
        features=features.astype(np.float32),
        labels=data["y"].astype(np.int64),
        subjects=data["subjects"].astype(np.int64),
        runs=data["runs"].astype(np.int64),
    )
    manifest = {
        "status": "completed",
        "dataset": "PhysioNetMI",
        "cache_path": str(feature_path),
        "source_cache": str(CANONICAL_CACHE),
        "contains_raw_mat_data": False,
        "contains_raw_edf_data": False,
        "contains_raw_trial_array": False,
        "contains_cbramod_features": True,
        "contains_labels": True,
        "features_shape": list(features.shape),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "feature_nan_count": int(np.isnan(features).sum()),
        "feature_inf_count": int(np.isinf(features).sum()),
        "labels_shape": list(data["y"].shape),
        "subjects_shape": list(data["subjects"].shape),
        "runs_shape": list(data["runs"].shape),
        "split_counts": {name: int(len(idx)) for name, idx in splits.items()},
        "resample_policy": "PhysioNetMI canonical [N,64,640] -> CBraMod [N,64,4,200]",
        "target_test_used_for_cache_extraction_only": True,
        "target_test_used_for_ranking_threshold_or_training": False,
        "checkpoint_audit": audit,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    write_json(manifest_path, manifest)
    write_json(OUT_DIR / "cbramod_feature_cache_summary.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
