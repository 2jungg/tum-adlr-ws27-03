"""Preview a random scene; --sample adds a panel with sampled configurations."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from metacosdf.scene_generator.generator import SceneGeneratorConfig, create_random_scene
from metacosdf.scene_generator.scene import SamplingConfig
from metacosdf.scene_generator.visualizer import SceneVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--bisection_steps", type=int, default=3)
    args = parser.parse_args()

    scene = create_random_scene(SceneGeneratorConfig(), np.random.default_rng())
    print(scene.obstacle)
    qs = torch.rand(4, scene.robot.dof) * (2.0 * math.pi)
    ground_truth = scene.cspace_sdf()
    visualizer = SceneVisualizer(scene=scene)

    n_panels = 3 if args.sample else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0))
    visualizer.plot_workspace(axes[0], qs=qs, title="workspace")
    visualizer.plot_cspace(ground_truth, axes[1], qs=qs, title="ground-truth C-space SDF [rad]")
    if args.sample:
        sampling = SamplingConfig(bisection_steps=args.bisection_steps)
        q_samples, y_samples = scene.sample_configurations(sampling)
        visualizer.plot_samples(q_samples, y_samples, axes[2], ground_truth=ground_truth)
    fig.tight_layout()
    plt.show()
