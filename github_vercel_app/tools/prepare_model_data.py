#!/usr/bin/env python
"""Prepare large optional model data that should not be committed to Git."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent

DATASETS_BASE = "data/kagglehub"
TRANSFERMARKT_DATASET = "davidcariboo/player-scores"
EXTERNAL_ELO_DATASET = "saifalnimri/international-football-elo-ratings"
SOFIFA_DATASETS = [
    "stefanoleone992/fifa-22-complete-player-dataset",
    "stefanoleone992/fifa-23-complete-player-dataset",
    "jmacd745/sofifa-data-set",
    "aniss7/fifa-player-data-from-sofifa-2025-06-03",
    "flynn28/eafc26-player-database",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download large model datasets when needed.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--include-sofifa", action="store_true")
    parser.add_argument("--include-external-elo", action="store_true")
    parser.add_argument("--force-sofifa-rebuild", action="store_true")
    return parser.parse_args()


def dataset_present(root: Path, owner_slug: str) -> bool:
    owner, slug = owner_slug.split("/", 1)
    return (root / DATASETS_BASE / "datasets" / owner / slug).exists()


def download_dataset(root: Path, owner_slug: str) -> None:
    if dataset_present(root, owner_slug):
        print(f"Dataset already present: {owner_slug}")
        return
    os.environ["KAGGLEHUB_CACHE"] = str(root / DATASETS_BASE)
    import kagglehub

    print(f"Downloading Kaggle dataset: {owner_slug}")
    path = kagglehub.dataset_download(owner_slug)
    print(f"Downloaded {owner_slug} -> {path}")


def run_python(root: Path, *args: str) -> None:
    command = [sys.executable, "-X", "utf8", *args]
    print("+", " ".join(command))
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = args.model_root.resolve()
    download_dataset(root, TRANSFERMARKT_DATASET)
    if args.include_external_elo:
        download_dataset(root, EXTERNAL_ELO_DATASET)
    if args.include_sofifa:
        for dataset in SOFIFA_DATASETS:
            download_dataset(root, dataset)
        output = root / "data" / "extracted" / "sofifa_yearly_player_ratings.csv"
        if args.force_sofifa_rebuild or not output.exists():
            run_python(root, "extract_sofifa_yearly_ratings.py")
        else:
            print(f"SoFIFA yearly ratings already present: {output}")


if __name__ == "__main__":
    main()
