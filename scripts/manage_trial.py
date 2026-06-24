#!/usr/bin/env python3
"""Small utility for creating and recording workbench trials."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / "workbench"


def slugify(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "trial"


def new_trial(name: str) -> Path:
    today = dt.datetime.now().strftime("%Y%m%d")
    slug = slugify(name)
    path = WORKBENCH / f"{today}_{slug}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "outputs").mkdir()
    payload = {
        "trial": slug,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "created",
        "promoted": False,
    }
    (path / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (path / "TRIAL.md").write_text(
        "\n".join(
            [
                f"# {slug}",
                "",
                "## Intent",
                "",
                "Describe the hypothesis and minimum success criteria.",
                "",
                "## Protocol",
                "",
                "Record dataset, split, backbone, augmentation, cert, training, and metrics.",
                "",
                "## Commands",
                "",
                "```bash",
                "# command goes here",
                "```",
                "",
                "## Results",
                "",
                "Fill this after the run.",
                "",
                "## Decision",
                "",
                "- `promote` / `archive` / `rerun`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "config.yaml").write_text("# Trial config placeholder\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="create a new workbench trial")
    new.add_argument("name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "new":
        path = new_trial(args.name)
        print(path)


if __name__ == "__main__":
    main()
