from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod paper reproduction gate with protocol separation.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out = workspace / "outputs" / "paper_reproduction"
    out.mkdir(parents=True, exist_ok=True)
    targets = read_csv(out / "paper_targets.csv")
    protocol_rows = [r for r in read_csv(out / "protocol_registry.csv") if r["model"] == "CBraMod"]
    step1 = read_json(workspace / "outputs" / "setup_audit_step1" / "compact_step1_result.json") if (workspace / "outputs" / "setup_audit_step1" / "compact_step1_result.json").exists() else {}

    metrics: List[Dict[str, object]] = []
    for protocol in protocol_rows:
        dataset = protocol["dataset"]
        if protocol["protocol_type"] == "executable_hybrid":
            ready = hybrid_ready(step1, dataset)
            metrics.append(row(protocol, "forward_smoke_only", "not_a_metric", "", "", "", "", "not_scored", "hybrid_forward_ready" if ready else "hybrid_forward_missing", protocol["notes"]))
        elif protocol["protocol_type"] == "code_exact":
            metrics.append(row(protocol, "official_code_default", "not_run", "", "", "", "", "not_run", "not_run_compute_guard", "Official code path exists but would launch full training on processed LMDB; not run in gate without exact processed-data audit."))
        else:
            for target in [t for t in targets if t["model"] == "CBraMod" and t["dataset"] == dataset and t["paper_protocol"] == "paper_exact"]:
                metrics.append(row(protocol, target["task"], target["metric"], target["paper_mean"], target["paper_std"], "", "", "failed", "needs_paper_preprocessing_lmdb", "Paper-exact full fine-tuning requires regenerated/audited LMDB and 50-epoch training; not substituted by frozen pooling or smoke."))

    loso_protocol = {
        "model": "CBraMod",
        "dataset": "BCIC-IV-2a LOSO key ingredients",
        "protocol_name": "paper_exact",
        "protocol_type": "paper_exact",
    }
    for target in [t for t in targets if t["model"] == "CBraMod" and "LOSO" in t["task"]]:
        metrics.append(row(loso_protocol, target["task"], target["metric"], target["paper_mean"], target["paper_std"], "", "", "failed", "not_run_missing_key_ingredients_implementation", "EA/session statistics/Mixup/subject-wise regularization implementation not identified in executable gate."))

    write_csv(out / "cbramod_metrics.csv", metrics)
    write_csv(out / "cbramod_discrepancy.csv", discrepancies(metrics))
    write_report(out / "cbramod_repro_report.md", metrics, protocol_rows)
    print(json.dumps({"status": "partial", "metrics_rows": len(metrics), "paper_exact_ran": False}, indent=2))


def hybrid_ready(step1: Dict[str, object], dataset: str) -> bool:
    ready = step1.get("ready_pairs", []) if isinstance(step1, dict) else []
    key = "CBraMod x PhysioNetMI" if dataset == "PhysioNet-MI" else "CBraMod x BCIC-IV-2a"
    return key in ready


def row(protocol: Dict[str, str], task: str, metric: str, paper_mean: object, paper_std: object, local_mean: object, local_std: object, status: str, likely_reason: str, notes: str) -> Dict[str, object]:
    return {
        "model": "CBraMod",
        "dataset": protocol["dataset"],
        "protocol_name": protocol["protocol_name"],
        "protocol_type": protocol["protocol_type"],
        "task": task,
        "metric": metric,
        "paper_mean": paper_mean,
        "paper_std": paper_std,
        "local_mean": local_mean,
        "local_std": local_std,
        "status": status,
        "likely_reason": likely_reason,
        "notes": notes,
    }


def discrepancies(metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for m in metrics:
        if not m["metric"] or m["metric"] in {"not_a_metric", "not_run"}:
            continue
        rows.append({
            "model": m["model"],
            "dataset": m["dataset"],
            "protocol_name": m["protocol_name"],
            "metric": m["metric"],
            "paper_mean": m["paper_mean"],
            "paper_std": m["paper_std"],
            "local_mean": m["local_mean"],
            "local_std": m["local_std"],
            "absolute_diff": "",
            "relative_diff": "",
            "within_2pp": False,
            "within_5pp": False,
            "status": m["status"],
            "likely_reason": m["likely_reason"],
        })
    return rows


def write_report(path: Path, metrics: List[Dict[str, object]], protocols: List[Dict[str, str]]) -> None:
    lines = [
        "# CBraMod Paper Reproduction Report",
        "",
        "## Protocol Separation",
        "",
    ]
    for protocol_type in ["code_exact", "paper_exact", "executable_hybrid"]:
        lines.append(f"### {protocol_type}")
        for p in [p for p in protocols if p["protocol_type"] == protocol_type]:
            lines.append(f"- {p['dataset']}: `{p['status']}`; entrypoint `{p['entrypoint']}`; {p['notes']}")
        lines.append("")
    lines.extend(["## Results", "", "| Dataset | Protocol | Task | Metric | Paper | Local | Status | Reason |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for m in metrics:
        paper = f"{m['paper_mean']}±{m['paper_std']}" if m["paper_mean"] != "" else ""
        local = f"{m['local_mean']}±{m['local_std']}" if m["local_mean"] != "" else ""
        lines.append(f"| {m['dataset']} | {m['protocol_name']} | {m['task']} | {m['metric']} | {paper} | {local} | {m['status']} | {m['likely_reason']} |")
    path.write_text("\n".join(lines) + "\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
