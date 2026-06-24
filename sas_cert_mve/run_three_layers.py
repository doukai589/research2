from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .augmentations import (
    AugmentedSample,
    generate_augmented_set,
    generate_mixed_candidate_pool,
)
from .bcic2a import (
    Standardizer,
    load_bcic2a_trials,
    make_few_shot_split,
    parse_subjects,
    records_to_arrays,
    select_records,
)
from .cert import score_candidates, scores_to_rows
from .metrics import roc_auc_binary, spearmanr
from .train_utils import clone_model, evaluate_model, make_model, set_seed, train_model


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.subjects = "1"
        args.seeds = "20"
        args.source_epochs = min(args.source_epochs, 1)
        args.finetune_epochs = min(args.finetune_epochs, 1)
        args.aug_per_trial = min(args.aug_per_trial, 2)
        args.out_dir = os.path.join(args.out_dir, "smoke")

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    subjects = parse_subjects(args.subjects)
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    cache_npz = args.cache_npz or os.path.join(args.out_dir, "cache", "bcic2a_trials.npz")

    records = load_bcic2a_trials(
        data_root=args.data_root,
        subjects=list(range(1, 10)),
        cache_npz=cache_npz,
        force_rebuild=args.force_rebuild_cache,
    )
    smoke_check(records)

    layer1_rows: List[Dict[str, object]] = []
    cert_rows: List[Dict[str, object]] = []
    layer3_rows: List[Dict[str, object]] = []

    for target_subject in subjects:
        for seed in seeds:
            print(f"[fold] target={target_subject} seed={seed}")
            fold = prepare_fold(records, target_subject, args.shot, seed)

            set_seed(seed)
            base_model = make_model(device)
            train_model(
                base_model,
                fold["source_x"],
                fold["source_y"],
                device=device,
                epochs=args.source_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            if args.run_layers in ("all", "layer1"):
                layer1_rows.extend(run_layer1(args, fold, base_model, device, target_subject, seed))

            candidates, scores = run_layer2(args, fold, base_model, device, target_subject, seed)
            if args.run_layers in ("all", "layer2"):
                for row in scores_to_rows(scores):
                    row.update({"target_subject": target_subject, "seed": seed})
                    cert_rows.append(row)

            if args.run_layers in ("all", "layer3"):
                layer3_rows.extend(run_layer3(args, fold, base_model, candidates, scores, device, target_subject, seed))

    if layer1_rows:
        write_csv(os.path.join(args.out_dir, "layer1_metrics.csv"), layer1_rows)
        write_json(os.path.join(args.out_dir, "layer1_summary.json"), summarize_layer1(layer1_rows))
    if cert_rows:
        write_csv(os.path.join(args.out_dir, "cert_scores.csv"), cert_rows)
        write_json(os.path.join(args.out_dir, "layer2_summary.json"), summarize_layer2(cert_rows))
    if layer3_rows:
        write_csv(os.path.join(args.out_dir, "layer3_metrics.csv"), layer3_rows)
        write_json(os.path.join(args.out_dir, "layer3_summary.json"), summarize_layer3(layer3_rows))

    print(f"[done] outputs written to {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-layer SAS-Cert-EEG MVE.")
    parser.add_argument(
        "--data-root",
        default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014",
    )
    parser.add_argument("--out-dir", default="sas_cert_mve_outputs")
    parser.add_argument("--cache-npz", default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="20,21,22")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--aug-per-trial", type=int, default=5)
    parser.add_argument("--source-epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-layers", choices=["all", "layer1", "layer2", "layer3"], default="all")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def smoke_check(records) -> None:
    a01_t = [r for r in records if r.subject == 1 and r.session == "T"]
    a01_e = [r for r in records if r.subject == 1 and r.session == "E"]
    if a01_t:
        assert a01_t[0].x.shape == (22, 800), f"Expected [22,800], got {a01_t[0].x.shape}"
        assert 0 <= a01_t[0].y <= 3, f"Expected label 0..3, got {a01_t[0].y}"
    if not a01_t or not a01_e:
        print("[warn] A01 T/E split incomplete in loaded subset; continuing for selected subjects.")
    else:
        print(f"[smoke] A01 train={len(a01_t)} eval={len(a01_e)} shape={a01_t[0].x.shape}")


def prepare_fold(records, target_subject: int, shot: int, seed: int) -> Dict[str, np.ndarray]:
    source_records, support_records, test_records = make_few_shot_split(records, target_subject, shot, seed)
    source_x, source_y, _, _, _ = records_to_arrays(source_records)
    support_x, support_y, _, _, support_ids = records_to_arrays(support_records)
    test_x, test_y, _, _, test_ids = records_to_arrays(test_records)
    standardizer = Standardizer.fit(source_x)
    source_x = standardizer.transform(source_x)
    support_x = standardizer.transform(support_x)
    test_x = standardizer.transform(test_x)
    return {
        "source_x": source_x,
        "source_y": source_y,
        "support_x": support_x,
        "support_y": support_y,
        "support_ids": support_ids,
        "test_x": test_x,
        "test_y": test_y,
        "test_ids": test_ids,
    }


def run_layer1(args, fold, base_model, device: str, target_subject: int, seed: int) -> List[Dict[str, object]]:
    rows = []
    rng = np.random.default_rng(seed + 1000 * target_subject)
    conditions = [
        ("NoAug", None, [0.0]),
        ("CleanAug", "clean", [0.5]),
        ("BadContent", "bad_content", [0.25, 0.5, 0.75]),
        ("BadStyle", "bad_style", [0.25, 0.5, 0.75]),
        ("BadPhysio", "bad_physio", [0.25, 0.5, 0.75]),
        ("BadArtifact", "bad_artifact", [0.25, 0.5, 0.75]),
    ]
    for condition, aug_type, intensities in conditions:
        for intensity in intensities:
            if args.smoke and condition not in ("NoAug", "CleanAug"):
                continue
            train_x, train_y = build_adaptation_data(
                fold,
                aug_type=aug_type,
                per_trial=args.aug_per_trial,
                rng=rng,
                intensity=intensity,
            )
            metrics = finetune_and_eval(args, base_model, train_x, train_y, fold, device)
            row = {
                "layer": "layer1",
                "target_subject": target_subject,
                "seed": seed,
                "condition": condition,
                "aug_type": aug_type or "none",
                "intensity": intensity,
            }
            row.update(metrics)
            rows.append(row)
            print(f"  [L1] {condition} intensity={intensity:.2f} acc={metrics['acc']:.4f}")
    return rows


def run_layer2(args, fold, base_model, device: str, target_subject: int, seed: int):
    rng = np.random.default_rng(seed + 2000 * target_subject)
    candidates = list(
        generate_mixed_candidate_pool(
            fold["support_x"],
            fold["support_y"],
            per_trial=args.aug_per_trial,
            rng=rng,
            style_bank=fold["source_x"],
            intensity=0.5,
        )
    )
    scores = score_candidates(candidates, fold["support_x"], base_model, device, batch_size=args.batch_size)
    labels = np.asarray([0 if s.is_bad else 1 for s in scores])
    totals = np.asarray([s.total for s in scores])
    auc = roc_auc_binary(labels, totals)
    print(f"  [L2] candidates={len(candidates)} clean-vs-bad AUC={auc:.4f}")
    return candidates, scores


def run_layer3(args, fold, base_model, candidates, scores, device: str, target_subject: int, seed: int):
    rows = []
    rng = np.random.default_rng(seed + 3000 * target_subject)
    candidate_x = np.stack([c.x for c in candidates]).astype(np.float32)
    candidate_y = np.asarray([c.y for c in candidates], dtype=np.int64)
    sorted_idx = np.argsort(np.asarray([s.total for s in scores]))[::-1]
    half = max(1, len(candidates) // 2)
    random_idx = rng.choice(len(candidates), size=half, replace=False)
    groups = {
        "NoAug": (fold["support_x"], fold["support_y"]),
        "NaiveAug": (
            np.concatenate([fold["support_x"], candidate_x], axis=0),
            np.concatenate([fold["support_y"], candidate_y], axis=0),
        ),
        "Random50": (
            np.concatenate([fold["support_x"], candidate_x[random_idx]], axis=0),
            np.concatenate([fold["support_y"], candidate_y[random_idx]], axis=0),
        ),
        "SASCertTop50": (
            np.concatenate([fold["support_x"], candidate_x[sorted_idx[:half]]], axis=0),
            np.concatenate([fold["support_y"], candidate_y[sorted_idx[:half]]], axis=0),
        ),
    }
    for group, (train_x, train_y) in groups.items():
        metrics = finetune_and_eval(args, base_model, train_x, train_y, fold, device)
        row = {
            "layer": "layer3",
            "target_subject": target_subject,
            "seed": seed,
            "group": group,
            "n_train": int(len(train_y)),
        }
        row.update(metrics)
        rows.append(row)
        print(f"  [L3] {group} acc={metrics['acc']:.4f} ece={metrics['ece']:.4f}")
    return rows


def build_adaptation_data(fold, aug_type, per_trial, rng, intensity):
    support_x = fold["support_x"]
    support_y = fold["support_y"]
    if aug_type is None:
        return support_x, support_y
    augmented = generate_augmented_set(
        support_x,
        support_y,
        aug_type=aug_type,
        per_trial=per_trial,
        rng=rng,
        intensity=intensity,
        style_bank=fold["source_x"],
    )
    aug_x = np.stack([a.x for a in augmented]).astype(np.float32)
    aug_y = np.asarray([a.y for a in augmented], dtype=np.int64)
    return np.concatenate([support_x, aug_x], axis=0), np.concatenate([support_y, aug_y], axis=0)


def finetune_and_eval(args, base_model, train_x, train_y, fold, device: str) -> Dict[str, float]:
    model = clone_model(base_model, device)
    train_model(
        model,
        train_x,
        train_y,
        device=device,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    metrics, _ = evaluate_model(model, fold["test_x"], fold["test_y"], device, batch_size=args.batch_size)
    return metrics


def summarize_layer1(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped = group_mean(rows, ["condition", "intensity"], ["acc", "macro_f1", "kappa", "ece", "nll"])
    clean_acc = _mean([r["acc"] for r in rows if r["condition"] == "CleanAug"])
    bad_high = [r for r in rows if str(r["condition"]).startswith("Bad") and float(r["intensity"]) == 0.75]
    bad_high_acc = _mean([r["acc"] for r in bad_high])
    trend_rows = [r for r in rows if str(r["condition"]).startswith("Bad")]
    rho = spearmanr(np.asarray([float(r["intensity"]) for r in trend_rows]), -np.asarray([float(r["acc"]) for r in trend_rows]))
    return {
        "grouped_mean": grouped,
        "clean_aug_acc_mean": clean_acc,
        "bad_high_acc_mean": bad_high_acc,
        "bad_high_minus_clean_acc": None if np.isnan(clean_acc) or np.isnan(bad_high_acc) else bad_high_acc - clean_acc,
        "bad_intensity_vs_accuracy_drop_spearman": rho,
    }


def summarize_layer2(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    labels = np.asarray([1 - int(r["is_bad"]) for r in rows])
    totals = np.asarray([float(r["total"]) for r in rows])
    auc = roc_auc_binary(labels, totals)
    sorted_rows = sorted(rows, key=lambda r: float(r["total"]), reverse=True)
    k = max(1, int(round(len(sorted_rows) * 0.3)))
    top = sorted_rows[:k]
    bottom = sorted_rows[-k:]
    return {
        "clean_vs_bad_auc": auc,
        "top30_bad_rate": _mean([int(r["is_bad"]) for r in top]),
        "bottom30_bad_rate": _mean([int(r["is_bad"]) for r in bottom]),
        "top30_mean_total": _mean([float(r["total"]) for r in top]),
        "bottom30_mean_total": _mean([float(r["total"]) for r in bottom]),
    }


def summarize_layer3(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped = group_mean(rows, ["group"], ["acc", "macro_f1", "kappa", "ece", "nll"])
    diffs = paired_diffs(rows, "SASCertTop50", ["NaiveAug", "Random50"], ["acc", "macro_f1", "ece"])
    return {"grouped_mean": grouped, "paired_diffs": diffs}


def paired_diffs(rows, reference_group: str, baselines: Sequence[str], metrics: Sequence[str]):
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["target_subject"], row["seed"])
        by_key[key][row["group"]] = row
    out = {}
    for baseline in baselines:
        values = {metric: [] for metric in metrics}
        for grouped in by_key.values():
            if reference_group not in grouped or baseline not in grouped:
                continue
            for metric in metrics:
                values[metric].append(float(grouped[reference_group][metric]) - float(grouped[baseline][metric]))
        out[f"{reference_group}_minus_{baseline}"] = {metric: _mean(vals) for metric, vals in values.items()}
    return out


def group_mean(rows: Sequence[Dict[str, object]], keys: Sequence[str], metrics: Sequence[str]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, values in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = len(values)
        for metric in metrics:
            item[metric] = _mean([float(v[metric]) for v in values])
        out.append(item)
    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(np.mean(values))


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
