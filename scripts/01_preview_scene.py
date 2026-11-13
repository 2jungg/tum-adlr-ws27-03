"""Preview a random 2-DoF scene: workspace, ground-truth C-space SDF, sampled configurations."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from metacosdf.scene_generator.generator import SceneGeneratorConfig, create_random_scene
from metacosdf.scene_generator.scene import SamplingConfig
from metacosdf.scene_generator.visualizer import SceneVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a randomly generated 2-DoF scene.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample configurations and visualize them in a 3rd subplot.",
    )
    parser.add_argument(
        "--bisection_steps",
        type=int,
        default=3,
        help="Iterative steps for boundary sampling (default: 3)",
    )
    args = parser.parse_args()

    scene = create_random_scene(SceneGeneratorConfig(), np.random.default_rng())
    print("Generated scene:", scene.obstacle)

    # a few random poses in [0, 2pi)^2 to inspect the collision coloring
    qs = torch.rand(4, scene.robot.dof) * (2.0 * math.pi)
    ground_truth = scene.cspace_sdf()
    visualizer = SceneVisualizer(scene=scene)

    n_panels = 3 if args.sample else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0))
    visualizer.plot_workspace(axes[0], qs=qs, title="workspace (green = free, red = collision)")
    visualizer.plot_cspace(ground_truth, axes[1], qs=qs, title="ground-truth C-space SDF [rad]")
    if args.sample:
        # Jacobian-weighted + boundary-densified samples
        sampling = SamplingConfig(bisection_steps=args.bisection_steps)
        q_samples, y_samples = scene.sample_configurations(sampling)
        visualizer.plot_samples(q_samples, y_samples, axes[2], ground_truth=ground_truth)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
