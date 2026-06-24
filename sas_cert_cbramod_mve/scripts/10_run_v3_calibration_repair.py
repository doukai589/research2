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
    parser = argparse.ArgumentParser(description="Run SAS-Cert-CBraMod v3 calibration repair MVE.")
    parser.add_argument("--project_root", default="/ai/224duibishiyan/615新研究/sas_cert_cbramod_mve")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--bcic2a_root", default="../CBraMod-main/tmp_in/BCIC2a/MNE-bnci-data/database/data-sets/001-2014")
    parser.add_argument("--score_variant", default="artifact_gate_content_rank")
    parser.add_argument("--artifact_reject_percentile", type=float, default=10.0)
    parser.add_argument("--w_min", type=float, default=0.2)
    parser.add_argument("--groups", nargs="+", default=[
        "NaiveAug",
        "SoftWeightArtifactReject_raw",
        "SWAR_TempScale_SourceVal",
        "SWAR_LabelSmoothing_eps005",
        "SWAR_LabelSmoothing_eps010",
        "SWAR_BrierMix_lam005",
        "SWAR_BrierMix_lam010",
        "ArtifactReject_raw",
    ])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="20,21,22")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--source_val_fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = bootstrap(args.project_root)

    from sas_cert.datasets.bcic2a_loader import load_bcic2a_trials, make_few_shot_split
    from sas_cert.training.linear_probe import LinearHead, clone_head
    from sas_cert.utils.io import ensure_dir, write_csv, write_json
    from sas_cert.utils.seed import set_seed

    outputs = project_root / "outputs"
    out_dir = ensure_dir(outputs / "layer3_v3")
    cal_dir = ensure_dir(outputs / "calibration_v3")
    file_audit = audit_files(outputs)
    write_json(out_dir / "file_audit.json", file_audit)
    if file_audit["blocked"]:
        write_blocked(outputs, file_audit["blocker"])
        raise RuntimeError(file_audit["blocker"])

    protocol = json.loads((outputs / "audit" / "protocol_audit.json").read_text())
    if protocol.get("protocol_leakage_detected"):
        write_blocked(outputs, "Existing protocol audit reports leakage.")
        raise RuntimeError("Existing protocol audit reports leakage.")

    subjects = [int(v) for v in args.subjects.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    workspace_root = Path(args.workspace_root).resolve()
    bcic2a_root = (workspace_root / args.bcic2a_root).resolve() if not Path(args.bcic2a_root).is_absolute() else Path(args.bcic2a_root)

    original = np.load(outputs / "features" / "original_features.npz", allow_pickle=False)
    features = original["features"].astype(np.float32)
    labels = original["labels"].astype(np.int64)
    trial_ids = original["trial_ids"].astype(str)
    id_to_idx = {tid: i for i, tid in enumerate(trial_ids.tolist())}

    records = load_bcic2a_trials(str(bcic2a_root), subjects=subjects)
    cert_by_fold = load_cert_by_fold(outputs / "layer2" / "cert_scores.csv")

    metrics_rows: List[Dict[str, object]] = []
    temp_rows: List[Dict[str, object]] = []
    cal_rows: List[Dict[str, object]] = []
    reject_fraction = args.artifact_reject_percentile / 100.0

    for target in subjects:
        for seed in seeds:
            print(f"[v3] target={target} seed={seed}", flush=True)
            set_seed(seed)
            source_records, support_records, test_records = make_few_shot_split(records, target, args.shot, seed)
            source_train_records, source_val_records = split_source_validation(source_records, args.source_val_fraction, seed + target * 997)
            source_train_idx = np.asarray([id_to_idx[r.trial_id] for r in source_train_records])
            source_val_idx = np.asarray([id_to_idx[r.trial_id] for r in source_val_records])
            support_idx = np.asarray([id_to_idx[r.trial_id] for r in support_records])
            test_idx = np.asarray([id_to_idx[r.trial_id] for r in test_records])

            source_x, source_y = features[source_train_idx], labels[source_train_idx]
            source_val_x, source_val_y = features[source_val_idx], labels[source_val_idx]
            support_x, support_y = features[support_idx], labels[support_idx]
            test_x, test_y = features[test_idx], labels[test_idx]

            fold_scores = cert_by_fold[(target, seed)]
            aug_cache = np.load(outputs / "features" / "augmented" / f"target{target}_seed{seed}.npz", allow_pickle=False)
            cand_x = aug_cache["features"].astype(np.float32)
            cand_y = aug_cache["labels"].astype(np.int64)
            if len(fold_scores) != len(cand_y):
                raise RuntimeError(f"Fold target={target} seed={seed} score/cache length mismatch.")

            variant_score = compute_variant_scores(fold_scores, args.score_variant)
            artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in fold_scores], dtype=float)
            reject_n = max(1, int(round(len(variant_score) * reject_fraction)))
            artifact_keep = np.ones(len(variant_score), dtype=bool)
            artifact_keep[np.argsort(artifact_risk)[::-1][:reject_n]] = False
            weights = args.w_min + (1.0 - args.w_min) * ranknorm(variant_score)

            base_head = LinearHead(200, 4)
            train_head_custom(base_head, source_x, source_y, device, args.source_epochs, args.batch_size, args.head_lr, args.weight_decay)

            for group in args.groups:
                train_x, train_y, sample_weight, label_smoothing, brier_lambda, use_temp, temp_value = make_group_data(
                    group, support_x, support_y, cand_x, cand_y, artifact_keep, weights
                )
                head = clone_head(base_head)
                train_head_custom(
                    head,
                    train_x,
                    train_y,
                    device,
                    args.finetune_epochs,
                    args.batch_size,
                    args.finetune_lr,
                    args.weight_decay,
                    sample_weight=sample_weight,
                    label_smoothing=label_smoothing,
                    brier_lambda=brier_lambda,
                )
                temperature = 1.0
                if use_temp:
                    temperature = fit_temperature_grid(head, source_val_x, source_val_y, device, args.batch_size)
                    temp_rows.append(
                        {
                            "target_subject": target,
                            "seed": seed,
                            "group": group,
                            "temperature": temperature,
                            "calibration_source": "source_validation",
                            "source_val_n": int(len(source_val_y)),
                        }
                    )
                metrics = evaluate_with_temperature(head, test_x, test_y, device, args.batch_size, temperature=temperature)
                metrics.update(
                    {
                        "target_subject": target,
                        "seed": seed,
                        "group": group,
                        "temperature": temperature,
                        "n_train": int(len(train_y)),
                        "source_val_n": int(len(source_val_y)),
                        "score_variant": args.score_variant,
                        "artifact_reject_percentile": args.artifact_reject_percentile,
                        "w_min": args.w_min,
                    }
                )
                metrics_rows.append(metrics)
                cal_rows.append(
                    {
                        "target_subject": target,
                        "seed": seed,
                        "group": group,
                        "temperature": temperature,
                        "ece": metrics["ece"],
                        "nll": metrics["nll"],
                        "brier": metrics["brier"],
                        "mean_confidence": metrics["mean_confidence"],
                    }
                )

    add_group_stability(metrics_rows)
    summary, paired, by_subject = summarize(metrics_rows)
    payload = {
        "status": "completed",
        "score_variant": args.score_variant,
        "main_method": "SoftWeightArtifactReject",
        "artifact_reject_percentile": args.artifact_reject_percentile,
        "w_min": args.w_min,
        "groups": args.groups,
        "summary": summary,
        "paired_comparison": paired,
        "protocol_leakage_detected": False,
        "backbone_frozen": True,
        "temperature_mean": mean([r["temperature"] for r in temp_rows]) if temp_rows else None,
        "temperature_std": float(np.std([r["temperature"] for r in temp_rows])) if temp_rows else None,
    }

    write_csv(out_dir / "layer3_v3_metrics.csv", metrics_rows)
    write_csv(out_dir / "paired_comparison_v3.csv", paired)
    write_csv(out_dir / "by_subject_table_v3.csv", by_subject)
    write_csv(out_dir / "calibration_table_v3.csv", cal_rows)
    write_csv(out_dir / "temperature_values.csv", temp_rows)
    write_json(out_dir / "layer3_v3_summary.json", payload)
    write_json(cal_dir / "temperature_summary.json", {"temperature_values": temp_rows, "temperature_mean": payload["temperature_mean"], "temperature_std": payload["temperature_std"]})
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def audit_files(outputs: Path) -> Dict[str, object]:
    required = [
        outputs / "layer2_v2" / "score_variant_auc.csv",
        outputs / "layer2_v2" / "layer2_v2_summary.json",
        outputs / "layer3_v2" / "layer3_v2_metrics.csv",
        outputs / "features" / "original_features.npz",
        outputs / "layer2" / "cert_scores.csv",
        outputs / "audit" / "protocol_audit.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    mat_count = len(list(outputs.parents[0].glob("**/*.mat")))
    blocker = ""
    if missing:
        blocker = f"Missing required files: {missing}"
    elif mat_count:
        blocker = f"Project contains copied .mat files: {mat_count}"
    return {
        "required_files": {str(p): p.exists() for p in required},
        "project_mat_file_count": mat_count,
        "blocked": bool(blocker),
        "blocker": blocker,
    }


def write_blocked(outputs: Path, blocker: str) -> None:
    (outputs / "BLOCKED_REPORT.md").write_text(f"# BLOCKED\n\n{blocker}\n")


def load_cert_by_fold(path: Path) -> Dict[Tuple[int, int], List[Dict[str, object]]]:
    out: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
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


def split_source_validation(records, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train = []
    val = []
    by_label = defaultdict(list)
    for record in records:
        by_label[record.y].append(record)
    for label_records in by_label.values():
        indices = np.arange(len(label_records))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * fraction)))
        val_set = set(indices[:n_val].tolist())
        for idx, record in enumerate(label_records):
            (val if idx in val_set else train).append(record)
    return train, val


def ranknorm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks.astype(np.float32)


def compute_variant_scores(rows: Sequence[Dict[str, object]], variant: str) -> np.ndarray:
    artifact = ranknorm([float(r["artifact_safe_score"]) for r in rows])
    content = ranknorm([float(r["content_score"]) for r in rows])
    style = ranknorm([float(r["style_score"]) for r in rows])
    physio = ranknorm([float(r["physio_score"]) for r in rows])
    artifact_risk = np.asarray([float(r["artifact_risk_raw"]) for r in rows], dtype=float)
    if variant == "artifact_gate_content_rank":
        score = content.copy()
        reject_n = max(1, int(round(len(score) * 0.10)))
        score[np.argsort(artifact_risk)[::-1][:reject_n]] = 0.0
        return score
    if variant == "artifact_content":
        return 0.55 * artifact + 0.45 * content
    if variant == "artifact_content_style":
        return 0.45 * artifact + 0.40 * content + 0.15 * style
    if variant == "artifact_content_physio_fixed":
        return 0.40 * artifact + 0.35 * content + 0.15 * physio + 0.10 * style
    if variant == "current_total":
        return 0.35 * content + 0.25 * physio + 0.25 * artifact + 0.15 * style
    raise ValueError(f"Unknown score variant: {variant}")


def make_group_data(group: str, support_x, support_y, cand_x, cand_y, artifact_keep, weights):
    if group == "NaiveAug":
        return np.concatenate([support_x, cand_x]), np.concatenate([support_y, cand_y]), None, 0.0, 0.0, False, 1.0
    if group == "ArtifactReject_raw":
        return np.concatenate([support_x, cand_x[artifact_keep]]), np.concatenate([support_y, cand_y[artifact_keep]]), None, 0.0, 0.0, False, 1.0

    x = np.concatenate([support_x, cand_x[artifact_keep]])
    y = np.concatenate([support_y, cand_y[artifact_keep]])
    w = np.concatenate([np.ones(len(support_y), dtype=np.float32), weights[artifact_keep].astype(np.float32)])
    if group == "SoftWeightArtifactReject_raw":
        return x, y, w, 0.0, 0.0, False, 1.0
    if group == "SWAR_TempScale_SourceVal":
        return x, y, w, 0.0, 0.0, True, 1.0
    if group == "SWAR_LabelSmoothing_eps005":
        return x, y, w, 0.05, 0.0, False, 1.0
    if group == "SWAR_LabelSmoothing_eps010":
        return x, y, w, 0.10, 0.0, False, 1.0
    if group == "SWAR_BrierMix_lam005":
        return x, y, w, 0.0, 0.05, False, 1.0
    if group == "SWAR_BrierMix_lam010":
        return x, y, w, 0.0, 0.10, False, 1.0
    raise ValueError(f"Unknown group: {group}")


def train_head_custom(
    head: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    sample_weight: Optional[np.ndarray] = None,
    label_smoothing: float = 0.0,
    brier_lambda: float = 0.0,
) -> torch.nn.Module:
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
            logits = head(xb)
            ce = smooth_ce_loss(logits, yb, label_smoothing)
            loss = (ce * wb).mean()
            if brier_lambda > 0:
                brier = brier_loss(logits, yb)
                loss = loss + brier_lambda * (brier * wb).mean()
            if torch.isnan(loss):
                raise RuntimeError("Head training loss became NaN.")
            loss.backward()
            opt.step()
    return head


def smooth_ce_loss(logits: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return torch.nn.functional.cross_entropy(logits, y, reduction="none")
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    n_classes = logits.shape[1]
    with torch.no_grad():
        target = torch.full_like(log_probs, eps / (n_classes - 1))
        target.scatter_(1, y.unsqueeze(1), 1.0 - eps)
    return -(target * log_probs).sum(dim=1)


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=1)
    target = torch.nn.functional.one_hot(y, num_classes=probs.shape[1]).float()
    return ((probs - target) ** 2).sum(dim=1)


def collect_logits(head: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    head.to(device)
    head.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()), batch_size=batch_size, shuffle=False)
    logits = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(head(xb.to(device)).detach().cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def fit_temperature_grid(head: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> float:
    logits, labels = collect_logits(head, x, y, device, batch_size)
    candidates = np.linspace(0.5, 5.0, 91)
    losses = [nll_from_logits(logits / temp, labels) for temp in candidates]
    return float(candidates[int(np.argmin(losses))])


def evaluate_with_temperature(head: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str, batch_size: int, temperature: float = 1.0) -> Dict[str, float]:
    logits, labels = collect_logits(head, x, y, device, batch_size)
    logits = logits / float(temperature)
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    return {
        "acc": float((pred == labels).mean()),
        "macro_f1": macro_f1(labels, pred, 4),
        "kappa": kappa_score(labels, pred, 4),
        "ece": ece_score(probs, labels, 10),
        "nll": nll_from_logits(logits, labels),
        "brier": brier_np(probs, labels, 4),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(softmax_np(logits), 1e-12, 1.0)
    return float(-np.log(probs[np.arange(len(y)), y]).mean())


def brier_np(probs: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    target = np.eye(n_classes, dtype=float)[y]
    return float(((probs - target) ** 2).sum(axis=1).mean())


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(acc[mask].mean() - conf[mask].mean()))
    return float(ece)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    vals = []
    for cls in range(n_classes):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def kappa_score(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def add_group_stability(rows: List[Dict[str, object]]) -> None:
    by_subject_group = defaultdict(list)
    for row in rows:
        by_subject_group[(row["target_subject"], row["group"])].append(row)
    group_subject_acc = defaultdict(list)
    for (_, group), vals in by_subject_group.items():
        group_subject_acc[group].append(mean(v["acc"] for v in vals))
    stability = {
        group: {
            "worst_subject_acc": float(np.min(vals)),
            "subject_wise_std": float(np.std(vals)),
        }
        for group, vals in group_subject_acc.items()
    }
    for row in rows:
        row.update(stability[row["group"]])


def summarize(rows: Sequence[Dict[str, object]]):
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

    by_subject_group = defaultdict(list)
    for row in rows:
        by_subject_group[(row["target_subject"], row["group"])].append(row)
    by_subject = []
    for (subject, group), vals in sorted(by_subject_group.items()):
        item = {"target_subject": subject, "group": group, "n": len(vals)}
        for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "mean_confidence"]:
            item[metric] = mean(v[metric] for v in vals)
        by_subject.append(item)

    baseline = {row["group"]: row for row in summary}["NaiveAug"]
    paired = []
    for row in summary:
        if row["group"] == "NaiveAug":
            continue
        delta = {metric: row[metric] - baseline[metric] for metric in ["acc", "macro_f1", "kappa", "ece", "nll", "brier", "worst_subject_acc", "subject_wise_std"]}
        passed_acc = delta["acc"] >= 0.01
        passed_f1 = delta["macro_f1"] >= 0.01
        passed_ece = delta["ece"] <= 0.01
        final = classify_group(row["group"], delta, passed_acc, passed_f1, passed_ece)
        paired.append(
            {
                "group": row["group"],
                "baseline_group": "NaiveAug",
                "delta_acc": delta["acc"],
                "delta_macro_f1": delta["macro_f1"],
                "delta_kappa": delta["kappa"],
                "delta_ece": delta["ece"],
                "delta_nll": delta["nll"],
                "delta_brier": delta["brier"],
                "delta_worst_subject_acc": delta["worst_subject_acc"],
                "delta_subject_wise_std": delta["subject_wise_std"],
                "passed_acc": passed_acc,
                "passed_f1": passed_f1,
                "passed_ece": passed_ece,
                "final_group_decision": final,
            }
        )
    return summary, paired, by_subject


def classify_group(group: str, delta: Dict[str, float], passed_acc: bool, passed_f1: bool, passed_ece: bool) -> str:
    if group == "ArtifactReject_raw":
        return "ARTIFACT_ONLY_BRANCH_DIAGNOSTIC" if (passed_acc or passed_f1) else "STOP"
    if (passed_acc or passed_f1) and passed_ece and delta["nll"] <= 0.01:
        return "GO_CALIBRATED_SOFTWEIGHT"
    if (passed_acc or passed_f1) and not passed_ece:
        return "CALIBRATION_STILL_FAIL"
    if passed_ece and not (passed_acc or passed_f1):
        return "CALIBRATION_OVERREGULARIZED"
    return "STOP"


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


if __name__ == "__main__":
    main()
