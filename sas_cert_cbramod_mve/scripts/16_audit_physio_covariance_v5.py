from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def load_physio_helpers(project_root: Path):
    path = project_root / "scripts" / "13_diagnose_physio_subscores.py"
    spec = importlib.util.spec_from_file_location("physio_v4_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Audit physio covariance construct-specific behavior.")
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
    hp = load_physio_helpers(project_root)

    from sas_cert.augmentations.ops import artifact_score_raw, clean_aug
    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.utils.io import ensure_dir, write_csv, write_json

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "physio_v5")
    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected") or len(list(project_root.glob("**/*.mat"))) > 0:
        (outputs / "BLOCKED_REPORT.md").write_text("# BLOCKED\n\nProtocol leakage or copied .mat detected.\n")
        raise RuntimeError("Protocol leakage or copied .mat detected.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    records = load_bcic2a_trials(str(bcic2a_root), subjects=args.subjects)
    rows: List[Dict[str, object]] = []
    for target in args.subjects:
        for seed in args.seeds:
            _, support_records, _ = make_few_shot_split(records, target, args.shot, seed)
            support_x = np.stack([r.x for r in support_records]).astype(np.float32)
            rng = np.random.default_rng(seed + target * 19000)
            for original_index, x in enumerate(support_x):
                for k in range(args.per_trial):
                    samples = [("clean", clean_aug(x, rng, 0.35).astype(np.float32))]
                    samples.extend(
                        [
                            ("BadPhysio_bandpower", hp.bad_physio_bandpower(x, rng, args.intensity).astype(np.float32)),
                            ("BadPhysio_covariance", hp.bad_physio_covariance(x, rng, args.intensity).astype(np.float32)),
                            ("BadPhysio_topology", hp.bad_physio_topology(x, rng, args.intensity).astype(np.float32)),
                            ("BadPhysio_channel_order", hp.bad_physio_channel_order(x, rng, args.intensity).astype(np.float32)),
                        ]
                    )
                    for aug_type, aug in samples:
                        rows.append(score_detectors(aug, x, target, seed, original_index, aug_type, f"{aug_type}_{original_index}_{k}", hp, artifact_score_raw))

    bad_types = ["BadPhysio_bandpower", "BadPhysio_covariance", "BadPhysio_topology", "BadPhysio_channel_order"]
    detectors = ["bandpower_score", "covariance_score", "topology_score", "channel_order_score", "content_score", "artifact_score"]
    matrix = []
    for bad_type in bad_types:
        subset = [r for r in rows if r["aug_type"] in {"clean", bad_type}]
        y = [1 if r["aug_type"] == "clean" else 0 for r in subset]
        item = {"bad_type": bad_type}
        for detector in detectors:
            item[detector] = hp.auc_score(y, [float(r[detector]) for r in subset])
        matrix.append(item)

    cov_row = next(r for r in matrix if r["bad_type"] == "BadPhysio_covariance")
    covariance_only_high = cov_row["covariance_score"] >= 0.70 and all(cov_row[k] < 0.70 for k in ["content_score", "artifact_score", "topology_score"])
    covariance_generalizes = any(
        row["bad_type"] in {"BadPhysio_topology", "BadPhysio_channel_order"} and row["covariance_score"] >= 0.70 for row in matrix
    )
    physio_detector_counts = {
        detector: sum(1 for row in matrix if row[detector] >= 0.70)
        for detector in ["bandpower_score", "covariance_score", "topology_score", "channel_order_score"]
    }
    physio_ready = any(count >= 2 for count in physio_detector_counts.values()) and not covariance_only_high
    audit = [
        {
            "covariance_auc_on_badphysio_covariance": cov_row["covariance_score"],
            "covariance_detector_is_construct_specific": bool(covariance_only_high),
            "covariance_generalizes_to_topology": bool(covariance_generalizes),
            "physio_ready_for_training": bool(physio_ready),
            "conclusion": "usable_for_specific_badphysio_covariance_only" if covariance_only_high else "cross_physio_signal_present",
        }
    ]
    summary = {
        "status": "completed",
        "physio_covariance_auc": cov_row["covariance_score"],
        "covariance_detector_is_construct_specific": bool(covariance_only_high),
        "covariance_generalizes_to_topology": bool(covariance_generalizes),
        "physio_ready_for_training": bool(physio_ready),
        "physio_detector_counts_auc_ge_070": physio_detector_counts,
        "conclusion": audit[0]["conclusion"],
        "protocol_leakage_detected": False,
    }
    write_csv(out_dir / "physio_cross_detection_auc_matrix.csv", matrix)
    write_csv(out_dir / "physio_covariance_same_source_audit.csv", audit)
    write_csv(out_dir / "physio_distribution_plots_data.csv", summarize_distribution(rows, detectors))
    write_json(out_dir / "physio_v5_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def score_detectors(x: np.ndarray, orig: np.ndarray, target: int, seed: int, original_index: int, aug_type: str, sample_id: str, hp, artifact_score_raw) -> Dict[str, object]:
    return {
        "target_subject": target,
        "seed": seed,
        "original_index": original_index,
        "sample_id": sample_id,
        "aug_type": aug_type,
        "is_bad": int(aug_type != "clean"),
        "bandpower_score": hp.bandpower_score(x, orig),
        "covariance_score": hp.covariance_score(x, orig),
        "topology_score": hp.topology_score(x, orig),
        "channel_order_score": hp.channel_order_score(x, orig),
        "content_score": content_score(x, orig),
        "artifact_score": -float(artifact_score_raw(x)),
    }


def content_score(x: np.ndarray, orig: np.ndarray) -> float:
    a = x.reshape(-1)
    b = orig.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def summarize_distribution(rows: Sequence[Dict[str, object]], detectors: Sequence[str]) -> List[Dict[str, object]]:
    out = []
    for aug_type in sorted({r["aug_type"] for r in rows}):
        subset = [r for r in rows if r["aug_type"] == aug_type]
        for detector in detectors:
            vals = np.asarray([float(r[detector]) for r in subset], dtype=float)
            out.append(
                {
                    "aug_type": aug_type,
                    "detector_score": detector,
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
