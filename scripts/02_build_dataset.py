"""Build train.pt, val.pt and eval.pt (format: docs/dataset.md)."""

import argparse
import time
from pathlib import Path

import torch

from metacosdf.config_utils import load_config
from metacosdf.dataset import DatasetConfig, build_split

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--eval", action="store_true", help="build only eval.pt")
    args = parser.parse_args()

    config = DatasetConfig.from_dict(load_config(args.config)["dataset"])
    out_dir = ROOT / config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["eval"] if args.eval else ["train", "val", "eval"]:
        start = time.perf_counter()
        path = out_dir / f"{split}.pt"
        torch.save(build_split(config, split), path)
        print(f"{path}: {config.n_scenes(split)} scenes, {time.perf_counter() - start:.1f} s")
