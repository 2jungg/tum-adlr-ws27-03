"""Preview a random scene; --sample adds a panel with sampled configurations."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from metacosdf.scene_generator.generator import SceneGeneratorConfig, create_random_scene
from metacosdf.scene_generator.scene import SamplingConfig
from metacosdf.visualization import plot_cspace, plot_samples, plot_workspace, save_figures

PLOTS_DIR = Path(__file__).parent.parent / "plots" / Path(__file__).stem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--bisection_steps", type=int, default=3)
    args = parser.parse_args()

    scene = create_random_scene(SceneGeneratorConfig(), np.random.default_rng())
    print(scene.obstacle)
    qs = torch.rand(4, scene.robot.dof) * (2.0 * math.pi)
    ground_truth = scene.cspace_sdf()

    n_panels = 3 if args.sample else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0))
    plot_workspace(scene, ax=axes[0], qs=qs, title="workspace")
    plot_cspace(ground_truth, ax=axes[1], qs=qs, title="ground-truth C-space SDF [rad]")
    if args.sample:
        sampling = SamplingConfig(bisection_steps=args.bisection_steps)
        q_samples, y_samples = scene.sample_configurations(sampling)
        plot_samples(q_samples, y_samples, ax=axes[2], ground_truth=ground_truth)
    fig.tight_layout()
    save_figures({"preview": fig}, PLOTS_DIR)
    plt.show()
