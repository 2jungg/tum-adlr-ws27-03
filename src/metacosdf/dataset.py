"""Datasets of random scenes with labeled configurations: train.pt, val.pt and eval.pt.

The file format is described in docs/dataset.md.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from metacosdf.config_utils import Config
from metacosdf.scene_generator.generator import SceneGeneratorConfig, create_random_scene
from metacosdf.scene_generator.scene import SamplingConfig

# every split has its own seed: config.seed + offset
SPLIT_SEED_OFFSETS = {"train": 0, "val": 10_000, "eval": 20_000}


@dataclass
class DatasetConfig(Config):
    generator: SceneGeneratorConfig = field(default_factory=SceneGeneratorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    n_train: int = 100
    n_val: int = 20
    n_eval: int = 20
    seed: int = 42
    # a relative path is relative to the repository root
    out_dir: str = "data/scenes"

    def n_scenes(self, split: str) -> int:
        return {"train": self.n_train, "val": self.n_val, "eval": self.n_eval}[split]


def build_split(config: DatasetConfig, split: str) -> dict[str, Any]:
    """
    Random scenes with labeled configurations for one split ("train", "val" or "eval").
    Only the eval split also stores each scene (Scene.to_dict()),
    so that its ground truth can be computed.
    """
    seed = config.seed + SPLIT_SEED_OFFSETS[split]
    rng = np.random.default_rng(seed)  # obstacles
    torch.manual_seed(seed)  # configurations: the sampling uses torch's global generator
    scenes = []
    for _ in range(config.n_scenes(split)):
        scene = create_random_scene(config.generator, rng)
        q, y = scene.sample_configurations(config.sampling)
        entry: dict[str, Any] = {"train_q": q, "train_y": y}
        if split == "eval":
            entry["metadata"] = scene.to_dict()
        scenes.append(entry)
    return {
        "config": {"split": split, "n_scenes": len(scenes), **config.to_dict()},
        "scenes": scenes,
    }
