"""Build the 2-DoF datasets train.pt, val.pt and eval.pt (format: docs/dataset.md)."""

import argparse
import time
from pathlib import Path

import torch

from metacosdf.config_utils import load_config
from metacosdf.dataset import DatasetConfig, build_split

ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 2-DoF datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dataset.yaml",
        help="YAML file with a 'dataset' section (default: configs/dataset.yaml).",
    )
    parser.add_argument("--eval", action="store_true", help="Build only eval.pt.")
    args = parser.parse_args()

    config = DatasetConfig.from_dict(load_config(args.config)["dataset"])
    out_dir = ROOT / config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["eval"] if args.eval else ["train", "val", "eval"]:
        if config.n_scenes(split) <= 0:
            continue
        print(f"[{split.upper()}] generating {config.n_scenes(split)} scenes ...")
        start = time.perf_counter()
        path = out_dir / f"{split}.pt"
        torch.save(build_split(config, split), path)
        size_mb = path.stat().st_size / 2**20
        elapsed = time.perf_counter() - start
        print(f"[{split.upper()}] saved {path} ({size_mb:.2f} MB, {elapsed:.1f} s)")


if __name__ == "__main__":
    main()
