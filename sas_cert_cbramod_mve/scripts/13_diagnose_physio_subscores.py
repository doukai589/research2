from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


SFREQ = 200.0


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose physio v4 subscore directions.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--bcic2a_root", default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22, 23, 24])
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--per_trial", type=int, default=2)
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.augmentations.ops import clean_aug
    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.utils.io import ensure_dir, write_csv, write_json

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "physio_v4")
    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected"):
        write_blocked(outputs, "Existing protocol audit reports leakage.")
        raise RuntimeError("Existing protocol audit reports leakage.")
    if len(list(project_root.glob("**/*.mat"))) > 0:
        write_blocked(outputs, "Raw .mat files detected inside project.")
        raise RuntimeError("Raw .mat files detected inside project.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    records = load_bcic2a_trials(str(bcic2a_root), subjects=args.subjects)

    rows: List[Dict[str, object]] = []
    for target in args.subjects:
        for seed in args.seeds:
            _, support_records, _ = make_few_shot_split(records, target, args.shot, seed)
            support_x = np.stack([r.x for r in support_records]).astype(np.float32)
            rng = np.random.default_rng(seed + target * 17000)
            for original_index, x in enumerate(support_x):
                for k in range(args.per_trial):
                    clean = clean_aug(x, rng, 0.35).astype(np.float32)
                    rows.append(score_sample(clean, x, target, seed, original_index, "clean", f"clean_{original_index}_{k}"))
                    for aug_type, fn in [
                        ("BadPhysio_bandpower", bad_physio_bandpower),
                        ("BadPhysio_covariance", bad_physio_covariance),
                        ("BadPhysio_topology", bad_physio_topology),
                        ("BadPhysio_channel_order", bad_physio_channel_order),
                    ]:
                        aug = fn(x, rng, args.intensity).astype(np.float32)
                        rows.append(score_sample(aug, x, target, seed, original_index, aug_type, f"{aug_type}_{original_index}_{k}"))

    auc_rows = []
    direction_rows = []
    component_map = {
        "BadPhysio_bandpower": "physio_bandpower",
        "BadPhysio_covariance": "physio_covariance",
        "BadPhysio_topology": "physio_topology",
        "BadPhysio_channel_order": "physio_channel_order",
    }
    for bad_type, component in component_map.items():
        subset = [r for r in rows if r["aug_type"] in {"clean", bad_type}]
        y = [1 if r["aug_type"] == "clean" else 0 for r in subset]
        scores = [float(r[component]) for r in subset]
        auc = auc_score(y, scores)
        item = {
            "bad_type": bad_type,
            "subscore": component,
            "original_auc": auc,
            "inverted_auc": 1.0 - auc if np.isfinite(auc) else float("nan"),
            "direction_maybe_wrong": bool(np.isfinite(auc) and auc < 0.5),
            "usable_physio_component": bool(np.isfinite(auc) and auc >= 0.70),
            "n": len(subset),
        }
        auc_rows.append(item)
        direction_rows.append(item.copy())

    dist_rows = summarize_distribution(rows)
    best = max(auc_rows, key=lambda r: r["original_auc"])
    summary = {
        "status": "completed",
        "physio_best_component": best["subscore"],
        "physio_best_component_auc": best["original_auc"],
        "physio_ready_for_training": bool(any(r["original_auc"] >= 0.70 for r in auc_rows) and not all(r["original_auc"] < 0.60 for r in auc_rows)),
        "physio_not_ready_for_training": bool(all(r["original_auc"] < 0.60 for r in auc_rows)),
        "usable_components": [r["subscore"] for r in auc_rows if r["original_auc"] >= 0.70],
        "direction_wrong_components": [r["subscore"] for r in auc_rows if r["original_auc"] < 0.5],
        "protocol_leakage_detected": False,
        "subscore_auc": auc_rows,
    }
    write_csv(out_dir / "physio_subscore_distribution.csv", dist_rows)
    write_csv(out_dir / "physio_subscore_auc.csv", auc_rows)
    write_csv(out_dir / "physio_direction_audit.csv", direction_rows)
    write_json(out_dir / "physio_v4_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def write_blocked(outputs: Path, blocker: str) -> None:
    (outputs / "BLOCKED_REPORT.md").write_text(f"# BLOCKED\n\n{blocker}\n")


def bad_physio_bandpower(x: np.ndarray, rng: np.random.Generator, intensity: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spec = np.fft.rfft(x, axis=1)
    mu = (freqs >= 8.0) & (freqs < 13.0)
    beta = (freqs >= 13.0) & (freqs < 30.0)
    spec[:, mu] *= 1.0 + rng.choice([-1, 1]) * (0.6 + intensity)
    spec[:, beta] *= 1.0 + rng.choice([-1, 1]) * (0.5 + 0.8 * intensity)
    return np.fft.irfft(spec, n=x.shape[1], axis=1)


def bad_physio_covariance(x: np.ndarray, rng: np.random.Generator, intensity: float) -> np.ndarray:
    mix = np.eye(x.shape[0]) + rng.normal(0, 0.04 + 0.10 * intensity, size=(x.shape[0], x.shape[0]))
    return mix @ x


def bad_physio_topology(x: np.ndarray, rng: np.random.Generator, intensity: float) -> np.ndarray:
    y = x.copy()
    n = max(2, int(round(x.shape[0] * (0.25 + 0.45 * intensity))))
    idx = rng.choice(x.shape[0], size=n, replace=False)
    shuffled = idx.copy()
    rng.shuffle(shuffled)
    y[idx] = x[shuffled]
    return y


def bad_physio_channel_order(x: np.ndarray, rng: np.random.Generator, intensity: float) -> np.ndarray:
    if rng.random() < 0.5:
        return x[::-1].copy()
    shift = int(rng.integers(3, max(4, x.shape[0] // 2)))
    return np.roll(x, shift=shift, axis=0)


def score_sample(x: np.ndarray, orig: np.ndarray, target: int, seed: int, original_index: int, aug_type: str, sample_id: str) -> Dict[str, object]:
    band = bandpower_score(x, orig)
    cov = covariance_score(x, orig)
    topo = topology_score(x, orig)
    order = channel_order_score(x, orig)
    return {
        "target_subject": target,
        "seed": seed,
        "original_index": original_index,
        "sample_id": sample_id,
        "aug_type": aug_type,
        "is_bad": int(aug_type != "clean"),
        "physio_bandpower": band,
        "physio_covariance": cov,
        "physio_topology": topo,
        "physio_channel_order": order,
    }


def bandpower_score(x: np.ndarray, orig: np.ndarray) -> float:
    bp_x = bandpowers(x)
    bp_o = bandpowers(orig)
    mu_dev = np.linalg.norm(bp_x[:, 0] - bp_o[:, 0])
    beta_dev = np.linalg.norm(bp_x[:, 1] - bp_o[:, 1])
    ratio_dev = np.linalg.norm((bp_x[:, 0] - bp_x[:, 1]) - (bp_o[:, 0] - bp_o[:, 1]))
    return float(-(mu_dev + beta_dev + ratio_dev))


def bandpowers(x: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / SFREQ)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    out = []
    for low, high in [(8, 13), (13, 30)]:
        mask = (freqs >= low) & (freqs < high)
        out.append(np.log(spec[:, mask].mean(axis=1) + 1e-8))
    return np.stack(out, axis=1)


def covariance_score(x: np.ndarray, orig: np.ndarray) -> float:
    return float(-np.linalg.norm(norm_cov(x) - norm_cov(orig), ord="fro"))


def norm_cov(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(1, x.shape[1] - 1)
    return cov / (np.trace(cov) + 1e-8)


def corr_mat(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.corrcoef(x), nan=0.0, posinf=0.0, neginf=0.0)


def topology_score(x: np.ndarray, orig: np.ndarray) -> float:
    cx = corr_mat(x)
    co = corr_mat(orig)
    drift = np.linalg.norm(cx - co, ord="fro")
    overlap = topk_overlap(cx, co, k=40)
    return float(-(drift + (1.0 - overlap)))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    iu = np.triu_indices_from(a, k=1)
    aa = np.abs(a[iu])
    bb = np.abs(b[iu])
    k = min(k, len(aa))
    sa = set(np.argsort(aa)[-k:].tolist())
    sb = set(np.argsort(bb)[-k:].tolist())
    return len(sa & sb) / max(1, k)


def channel_order_score(x: np.ndarray, orig: np.ndarray) -> float:
    smooth_x = np.diff(x, axis=0)
    smooth_o = np.diff(orig, axis=0)
    smooth_dev = np.linalg.norm(smooth_x - smooth_o) / (np.linalg.norm(smooth_o) + 1e-8)
    corr_diag = np.mean([np.corrcoef(x[i], orig[i])[0, 1] for i in range(x.shape[0])])
    return float(-(smooth_dev + (1.0 - np.nan_to_num(corr_diag))))


def auc_score(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize_distribution(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for aug_type in sorted({r["aug_type"] for r in rows}):
        subset = [r for r in rows if r["aug_type"] == aug_type]
        for subscore in ["physio_bandpower", "physio_covariance", "physio_topology", "physio_channel_order"]:
            vals = np.asarray([float(r[subscore]) for r in subset])
            out.append(
                {
                    "aug_type": aug_type,
                    "subscore": subscore,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            )
    return out


if __name__ == "__main__":
    main()
