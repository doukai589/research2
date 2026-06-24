from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def bootstrap(project_root: str) -> Path:
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Run v4 confirmatory MVE for calibrated SAS-Cert-CBraMod.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--bcic2a_root", default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
    parser.add_argument("--main_method", default="SASCert_SoftAR_LS010")
    parser.add_argument("--score_variant", default="artifact_gate_content_rank")
    parser.add_argument("--artifact_reject_percentile", type=float, default=10.0)
    parser.add_argument("--w_min", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22, 23, 24])
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.augmentations.ops import AugmentedSample, augment_one
    from sas_cert.backbones.cbramod_wrapper import CBraModFeatureExtractor
    from sas_cert.cert.sas_cert_lite import score_candidates
    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.training.linear_probe import LinearHead, clone_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "layer3_v4")
    audit_dir = ensure_dir(outputs / "v4_audit")
    audit = file_audit(project_root, outputs)
    write_json(audit_dir / "confirmatory_file_audit.json", audit)
    if audit["blocked"]:
        write_blocked(outputs, audit["blocker"])
        raise RuntimeError(audit["blocker"])

    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected"):
        write_blocked(outputs, "Existing protocol audit reports leakage.")
        raise RuntimeError("Existing protocol audit reports leakage.")

    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    original = np.load(outputs / "features" / "original_features.npz", allow_pickle=False)
    features = original["features"].astype(np.float32)
    labels = original["labels"].astype(np.int64)
    trial_ids = original["trial_ids"].astype(str)
    id_to_idx = {tid: i for i, tid in enumerate(trial_ids.tolist())}
    records = load_bcic2a_trials(str(bcic2a_root), subjects=args.subjects)

    cert_by_fold = load_existing_cert(outputs / "layer2" / "cert_scores.csv")
    needs_extractor = any(
        not (outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz").exists() or (target, seed) not in cert_by_fold
        for target in args.subjects
        for seed in args.seeds
    )
    extractor = CBraModFeatureExtractor(str(project_root / "third_party" / "CBraMod"), device=device) if needs_extractor else None
    metrics_rows: List[Dict[str, object]] = []
    reject_fraction = args.artifact_reject_percentile / 100.0

    for target in args.subjects:
        for seed in args.seeds:
            print(f"[v4-confirm] target={target} seed={seed}", flush=True)
            set_seed(seed)
            source_records, support_records, test_records = make_few_shot_split(records, target, args.shot, seed)
            source_idx = np.asarray([id_to_idx[r.trial_id] for r in source_records])
            support_idx = np.asarray([id_to_idx[r.trial_id] for r in support_records])
            test_idx = np.asarray([id_to_idx[r.trial_id] for r in test_records])
            source_x_feat, source_y = features[source_idx], labels[source_idx]
            support_x_feat, support_y = features[support_idx], labels[support_idx]
            test_x_feat, test_y = features[test_idx], labels[test_idx]
            support_raw = np.stack([r.x for r in support_records]).astype(np.float32)
            source_raw = np.stack([r.x for r in source_records]).astype(np.float32)

            cand_feat, cand_y, fold_scores = get_candidate_pool(
                outputs,
                cert_by_fold,
                target,
                seed,
                support_raw,
                support_y,
                support_x_feat,
                source_raw,
                args,
                extractor,
                device,
                project_root,
                score_candidates,
                AugmentedSample,
                augment_one,
            )
            score = compute_variant_scores(fold_scores, args.score_variant)
            artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in fold_scores], dtype=float)
            reject_n = max(1, int(round(len(score) * reject_fraction)))
            keep = np.ones(len(score), dtype=bool)
            keep[np.argsort(artifact_risk)[::-1][:reject_n]] = False
            sw = args.w_min + (1.0 - args.w_min) * ranknorm(score)

            base_head = LinearHead(200, 4)
            train_head(base_head, source_x_feat, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)

            groups = {
                "NaiveAug": (np.concatenate([support_x_feat, cand_feat]), np.concatenate([support_y, cand_y]), None, 0.0),
                "SASCert_SoftAR_LS010": (
                    np.concatenate([support_x_feat, cand_feat[keep]]),
                    np.concatenate([support_y, cand_y[keep]]),
                    np.concatenate([np.ones(len(support_y), dtype=np.float32), sw[keep].astype(np.float32)]),
                    args.label_smoothing,
                ),
                "ArtifactReject_raw_diagnostic": (
                    np.concatenate([support_x_feat, cand_feat[keep]]),
                    np.concatenate([support_y, cand_y[keep]]),
                    None,
                    0.0,
                ),
            }
            for group, (train_x, train_y, weights, eps) in groups.items():
                head = clone_head(base_head)
                train_head(head, train_x, train_y, device, args.finetune_epochs, args.batch_size, args.finetune_lr, args.weight_decay, sample_weight=weights, label_smoothing=eps)
                row = evaluate(head, test_x_feat, test_y, device, args.batch_size)
                row.update(
                    {
                        "target_subject": target,
                        "seed": seed,
                        "group": group,
                        "n_train": int(len(train_y)),
                        "score_variant": args.score_variant,
                        "artifact_reject_percentile": args.artifact_reject_percentile,
                        "w_min": args.w_min,
                        "label_smoothing": eps,
                    }
                )
                metrics_rows.append(row)

    add_group_stability(metrics_rows)
    summary, paired, by_subject, win_rate = summarize(metrics_rows)
    payload = {
        "status": "completed",
        "main_method": args.main_method,
        "score_variant": args.score_variant,
        "artifact_reject_percentile": args.artifact_reject_percentile,
        "w_min": args.w_min,
        "label_smoothing": args.label_smoothing,
        "seeds": args.seeds,
        "summary": summary,
        "paired_comparison": paired,
        "protocol_leakage_detected": False,
        "backbone_frozen": True,
    }
    write_csv(out_dir / "layer3_v4_metrics.csv", metrics_rows)
    write_csv(out_dir / "paired_comparison_v4.csv", paired)
    write_csv(out_dir / "by_subject_table_v4.csv", by_subject)
    write_csv(out_dir / "win_rate_table_v4.csv", win_rate)
    write_json(out_dir / "layer3_v4_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def file_audit(project_root: Path, outputs: Path) -> Dict[str, object]:
    required = [
        outputs / "features" / "original_features.npz",
        outputs / "layer2" / "cert_scores.csv",
        outputs / "audit" / "protocol_audit.json",
        project_root / "third_party" / "CBraMod" / "models" / "cbramod.py",
        project_root / "third_party" / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
    ]
    missing = [str(p) for p in required if not p.exists()]
    mat_count = len(list(project_root.glob("**/*.mat")))
    blocker = ""
    if missing:
        blocker = f"Missing required files: {missing}"
    elif mat_count:
        blocker = f"Raw .mat files detected inside project: {mat_count}"
    return {"required_files": {str(p): p.exists() for p in required}, "project_mat_file_count": mat_count, "blocked": bool(blocker), "blocker": blocker}


def write_blocked(outputs: Path, blocker: str) -> None:
    (outputs / "BLOCKED_REPORT.md").write_text(f"# BLOCKED\n\n{blocker}\n")


def load_existing_cert(path: Path) -> Dict[Tuple[int, int], List[Dict[str, object]]]:
    out: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    if not path.exists():
        return out
    numeric = ["sas_score", "artifact_risk_raw", "artifact_safe_score", "content_score", "style_score", "physio_score"]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["target_subject"] = int(row["target_subject"])
            row["seed"] = int(row["seed"])
            row["is_bad"] = int(row["is_bad"])
            row["label"] = int(row["label"])
            for key in numeric:
                row[key] = float(row[key])
            out[(row["target_subject"], row["seed"])].append(row)
    return out


def get_candidate_pool(outputs, cert_by_fold, target, seed, support_raw, support_y, support_feat, source_raw, args, extractor, device, project_root, score_candidates, AugmentedSample, augment_one):
    cache_path = outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz"
    if cache_path.exists() and (target, seed) in cert_by_fold:
        cache = np.load(cache_path, allow_pickle=False)
        return cache["features"].astype(np.float32), cache["labels"].astype(np.int64), cert_by_fold[(target, seed)]
    if extractor is None:
        from sas_cert.backbones.cbramod_wrapper import CBraModFeatureExtractor

        extractor = CBraModFeatureExtractor(str(project_root / "third_party" / "CBraMod"), device=device)
    rng = np.random.default_rng(seed + target * 2000)
    candidates = []
    for i in range(support_raw.shape[0]):
        for k in range(args.n_aug):
            x_aug = augment_one(support_raw[i], "clean", rng, 0.35, source_raw)
            candidates.append(AugmentedSample(x_aug.astype(np.float32), int(support_y[i]), i, "clean", 0.35, False, f"clean_{i:04d}_{k:02d}"))
        for aug_type in ["bad_content", "bad_style", "bad_physio", "bad_artifact"]:
            x_aug = augment_one(support_raw[i], aug_type, rng, args.intensity, source_raw)
            candidates.append(AugmentedSample(x_aug.astype(np.float32), int(support_y[i]), i, aug_type, args.intensity, True, f"{aug_type}_{i:04d}_00"))
    cand_x = np.stack([c.x for c in candidates]).astype(np.float32)
    cand_y = np.asarray([c.y for c in candidates], dtype=np.int64)
    cand_feat, _ = extractor.extract(cand_x, batch_size=args.feature_batch_size)
    fold_scores = score_candidates(candidates, support_raw, support_feat, cand_feat)
    np.savez_compressed(
        cache_path,
        features=cand_feat.astype(np.float32),
        labels=cand_y,
        aug_type=np.asarray([c.aug_type for c in candidates]),
        is_bad=np.asarray([int(c.is_bad) for c in candidates], dtype=np.int64),
        original_index=np.asarray([c.original_index for c in candidates], dtype=np.int64),
    )
    return cand_feat.astype(np.float32), cand_y, fold_scores


def ranknorm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks.astype(np.float32)


def compute_variant_scores(rows: Sequence[Dict[str, object]], variant: str) -> np.ndarray:
    content = ranknorm([float(r["content_score"]) for r in rows])
    artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in rows], dtype=float)
    if variant != "artifact_gate_content_rank":
        raise ValueError(f"v4 is fixed to artifact_gate_content_rank, got {variant}")
    score = content.copy()
    reject_n = max(1, int(round(len(score) * 0.10)))
    score[np.argsort(artifact_risk)[::-1][:reject_n]] = 0.0
    return score


def train_head(head, x, y, device, epochs, batch_size, lr, weight_decay, sample_weight: Optional[np.ndarray] = None, label_smoothing: float = 0.0):
    head.to(device)
    head.train()
    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).long()
    tw = torch.ones(len(y), dtype=torch.float32) if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))
    loader = DataLoader(TensorDataset(tx, ty, tw), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (smooth_ce(head(xb), yb, label_smoothing) * wb).mean()
            if torch.isnan(loss):
                raise RuntimeError("Head loss became NaN.")
            loss.backward()
            opt.step()


def smooth_ce(logits, y, eps: float):
    if eps <= 0:
        return torch.nn.functional.cross_entropy(logits, y, reduction="none")
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    target = torch.full_like(log_probs, eps / (logits.shape[1] - 1))
    target.scatter_(1, y.unsqueeze(1), 1.0 - eps)
    return -(target * log_probs).sum(dim=1)


def collect_logits(head, x, y, device, batch_size):
    head.to(device)
    head.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()), batch_size=batch_size, shuffle=False)
    logits = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(head(xb.to(device)).detach().cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def evaluate(head, x, y, device, batch_size):
    logits, labels = collect_logits(head, x, y, device, batch_size)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    return {
        "acc": float((pred == labels).mean()),
        "macro_f1": macro_f1(labels, pred, 4),
        "kappa": kappa(labels, pred, 4),
        "ece": ece(probs, labels),
        "nll": nll(logits, labels),
        "brier": brier(probs, labels, 4),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, y):
    probs = np.clip(softmax(logits), 1e-12, 1.0)
    return float(-np.log(probs[np.arange(len(y)), y]).mean())


def brier(probs, y, n_classes):
    target = np.eye(n_classes)[y]
    return float(((probs - target) ** 2).sum(axis=1).mean())


def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return float(out)


def macro_f1(y, pred, n_classes):
    vals = []
    for cls in range(n_classes):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def kappa(y, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    total = cm.sum()
    po = np.trace(cm) / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def mean(vals):
    vals = [float(v) for v in vals]
    return float(np.mean(vals)) if vals else float("nan")


def add_group_stability(rows):
    by_subject_group = defaultdict(list)
    for row in rows:
        by_subject_group[(row["target_subject"], row["group"])].append(row)
    group_subject_acc = defaultdict(list)
    for (_, group), vals in by_subject_group.items():
        group_subject_acc[group].append(mean(v["acc"] for v in vals))
    stability = {g: {"worst_subject_acc": float(np.min(v)), "subject_wise_std": float(np.std(v))} for g, v in group_subject_acc.items()}
    for row in rows:
        row.update(stability[row["group"]])


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary = []
    for group, vals in sorted(grouped.items()):
        item = {"group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "mean_confidence"]:
            item[metric] = mean(v[metric] for v in vals)
        item["worst_subject_acc"] = vals[0]["worst_subject_acc"]
        item["subject_wise_std"] = vals[0]["subject_wise_std"]
        summary.append(item)
    baseline = next(row for row in summary if row["group"] == "NaiveAug")
    paired = []
    for row in summary:
        if row["group"] == "NaiveAug":
            continue
        delta = {m: row[m] - baseline[m] for m in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "worst_subject_acc", "subject_wise_std"]}
        passed = {
            "passed_macro_f1": delta["macro_f1"] >= 0.01,
            "passed_acc": delta["acc"] >= 0.005,
            "passed_ece": delta["ece"] <= 0.01,
            "passed_nll": delta["nll"] <= 0.005,
            "passed_brier": delta["brier"] <= 0.005,
        }
        if row["group"] == "SASCert_SoftAR_LS010":
            if all(passed.values()) and delta["subject_wise_std"] <= baseline["subject_wise_std"] * 0.05 and delta["worst_subject_acc"] >= -0.01:
                decision = "GO_STABLE_CALIBRATED_SOFTWEIGHT"
            elif passed["passed_macro_f1"] and not passed["passed_acc"]:
                decision = "PARTIAL_GO_F1_STABLE"
            elif not passed["passed_ece"]:
                decision = "CALIBRATION_REGRESSION"
            else:
                decision = "EFFECT_NOT_STABLE"
        else:
            decision = "ARTIFACT_ONLY_BRANCH_DIAGNOSTIC"
        paired.append({"group": row["group"], "baseline_group": "NaiveAug", **{f"delta_{k}": v for k, v in delta.items()}, **passed, "final_group_decision": decision})
    by_subject = []
    by_subject_group = defaultdict(list)
    for row in rows:
        by_subject_group[(row["target_subject"], row["group"])].append(row)
    for (subject, group), vals in sorted(by_subject_group.items()):
        item = {"target_subject": subject, "group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "mean_confidence"]:
            item[metric] = mean(v[metric] for v in vals)
        by_subject.append(item)
    win_rate = compute_win_rates(by_subject)
    win_map = {row["group"]: row for row in win_rate}
    for row in paired:
        row["subject_win_rate_acc"] = win_map[row["group"]]["subject_win_rate_acc"]
        row["subject_win_rate_macro_f1"] = win_map[row["group"]]["subject_win_rate_macro_f1"]
    return summary, paired, by_subject, win_rate


def compute_win_rates(by_subject):
    by_subject_map = defaultdict(dict)
    for row in by_subject:
        by_subject_map[row["target_subject"]][row["group"]] = row
    out = []
    groups = sorted({row["group"] for row in by_subject if row["group"] != "NaiveAug"})
    for group in groups:
        acc_wins = []
        f1_wins = []
        for vals in by_subject_map.values():
            if group in vals and "NaiveAug" in vals:
                acc_wins.append(float(vals[group]["acc"]) > float(vals["NaiveAug"]["acc"]))
                f1_wins.append(float(vals[group]["macro_f1"]) > float(vals["NaiveAug"]["macro_f1"]))
        out.append({"group": group, "baseline_group": "NaiveAug", "subject_win_rate_acc": mean(acc_wins), "subject_win_rate_macro_f1": mean(f1_wins), "n_subjects": len(acc_wins)})
    return out


if __name__ == "__main__":
    main()
