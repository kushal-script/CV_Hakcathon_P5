#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
TEMPLATE = EXP_DIR / "_template"
INDEX = EXP_DIR / "INDEX.md"
TYPES = [
    "data", "classical", "deep", "detection", "hybrid", "ablation",
    "robustness", "optimisation", "calibration", "failure-analysis",
]
TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".txt"}


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"


def unique_timestamp_dir() -> tuple[str, Path]:
    while True:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = EXP_DIR / ts
        if not path.exists():
            return ts, path
        time.sleep(1)


def main() -> int:
    p = argparse.ArgumentParser(description="Create a timestamped experiment folder")
    p.add_argument("--title", required=True)
    p.add_argument("--type", required=True, choices=TYPES)
    p.add_argument("--author", default="team")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    if not TEMPLATE.is_dir():
        print(f"Template folder missing: {TEMPLATE}", file=sys.stderr)
        return 1

    ts, path = unique_timestamp_dir()
    shutil.copytree(TEMPLATE, path)
    for sub in ("logs", "figures", "artifacts"):
        (path / sub).mkdir(exist_ok=True)
        (path / sub / ".gitkeep").touch()

    fields = {
        "{{ID}}": ts,
        "{{TITLE}}": args.title,
        "{{TYPE}}": args.type,
        "{{AUTHOR}}": args.author,
        "{{STARTED}}": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "{{COMMIT}}": git_commit(),
    }
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in TEXT_SUFFIXES:
            text = f.read_text()
            for k, v in fields.items():
                text = text.replace(k, v)
            f.write_text(text)

    title = args.title.replace("|", "/")
    notes = args.notes.replace("|", "/")
    row = f"| [{ts}]({ts}/README.md) | {title} | {args.type} | {args.author} | running | val | | {notes} |\n"
    with INDEX.open("a") as fh:
        fh.write(row)

    print(f"Created {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
