#!/usr/bin/env python
"""Download the latest martj42 international_results CSV files with a local backup."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import requests


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent
BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update international_results CSVs.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.model_root.resolve() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = data_dir / "backups" / f"international_results_{datetime.now():%Y%m%d_%H%M%S}"
    if not args.no_backup:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            source = data_dir / name
            if source.exists():
                shutil.copy2(source, backup_dir / name)

    for name in FILES:
        url = f"{BASE_URL}/{name}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (data_dir / name).write_bytes(response.content)
        print(f"Updated {name} from {url}")

    if not args.no_backup:
        print(f"Backup: {backup_dir}")


if __name__ == "__main__":
    main()
