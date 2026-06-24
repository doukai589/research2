from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIRepNet paper reproduction gate with protocol separation.")
    parser.add_argument("--workspace_root", default="/ai/224duibishiyan/615新研究")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace_root).resolve()
    out = workspace / "outputs" / "paper_reproduction"
    out.mkdir(parents=True, exist_ok=True)
    targets = read_csv(out / "paper_targets.csv")
    protocols = [r for r in read_csv(out / "protocol_registry.csv") if r["model"] == "MIRepNet"]
    adapter = read_json(out / "mirepnet_adapter_manifest.json") if (out / "mirepnet_adapter_manifest.json").exists() else {"adapter_ready": False}

    rows: List[Dict[str, object]] = []
    for protocol in protocols:
        if protocol["protocol_type"] == "code_exact":
            rows.append(row(protocol, "official_code_default", "not_run", "", "", "", "", "failed", "official_repo_behavior_unclear", "Official README only documents BNCI2014004; target dataset code/preprocessed arrays absent."))
        elif protocol["protocol_type"] == "executable_hybrid":
            rows.append(row(protocol, "adapter_hybrid", "not_run", "", "", "", "", "failed", "adapter_needed", "Hybrid intentionally not created until adapter is validated."))
        else:
            for target in [t for t in targets if t["model"] == "MIRepNet" and t["dataset"] == protocol["dataset"]]:
                rows.append(row(protocol, target["task"], target["metric"], target["paper_mean"], target["paper_std"], "", "", "failed", "adapter_needed", "Paper-exact run blocked because MIRepNet adapter is not ready; no 45-channel X.npy/labels.npy generated."))

    write_csv(out / "mirepnet_metrics.csv", rows)
    write_csv(out / "mirepnet_discrepancy.csv", discrepancies(rows))
    write_report(out / "mirepnet_repro_report.md", rows, protocols, adapter)
    print(json.dumps({"status": "partial", "adapter_ready": adapter.get("adapter_ready"), "paper_exact_ran": False}, indent=2))


def row(protocol: Dict[str, str], task: str, metric: str, paper_mean: object, paper_std: object, local_mean: object, local_std: object, status: str, likely_reason: str, notes: str) -> Dict[str, object]:
    return {
        "model": "MIRepNet",
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
        if not m["metric"] or m["metric"] == "not_run":
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


def write_report(path: Path, rows: List[Dict[str, object]], protocols: List[Dict[str, str]], adapter: Dict[str, object]) -> None:
    lines = ["# MIRepNet Paper Reproduction Report", "", "## Adapter", "", f"- adapter_ready: `{adapter.get('adapter_ready')}`", f"- reason: {adapter.get('reason', '')}", ""]
    lines.extend(["## Protocol Separation", ""])
    for protocol_type in ["code_exact", "paper_exact", "executable_hybrid"]:
        lines.append(f"### {protocol_type}")
        for p in [p for p in protocols if p["protocol_type"] == protocol_type]:
            lines.append(f"- {p['dataset']}: `{p['status']}`; entrypoint `{p['entrypoint']}`; {p['notes']}")
        lines.append("")
    lines.extend(["## Results", "", "| Dataset | Protocol | Task | Metric | Paper | Local | Status | Reason |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for r in rows:
        paper = f"{r['paper_mean']}±{r['paper_std']}" if r["paper_mean"] != "" else ""
        local = f"{r['local_mean']}±{r['local_std']}" if r["local_mean"] != "" else ""
        lines.append(f"| {r['dataset']} | {r['protocol_name']} | {r['task']} | {r['metric']} | {paper} | {local} | {r['status']} | {r['likely_reason']} |")
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
