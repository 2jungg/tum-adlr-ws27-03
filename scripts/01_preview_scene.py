"""Preview a randomly generated 2-DoF scene in workspace and C-space."""

import argparse
import math
import torch

from metacosdf.scene_generator.arm import PlanarArm
from metacosdf.scene_generator.generator import create_random_scene
from metacosdf.scene_generator.visualizer import SceneVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a randomly generated 2-DoF scene.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample configurations and visualize them in a 3rd subplot.",
    )
    args = parser.parse_args()

    arm = PlanarArm()
    scene = create_random_scene(arm)

    print(f"Generated scene with {len(scene.obstacles)} obstacle(s):")
    for idx, obs in enumerate(scene.obstacles, start=1):
        print(f"  {idx}. {obs.to_dict()}")

    # Sample a few random poses in [0, 2pi)^2 to inspect collision coloring
    qs = torch.rand(4, arm.dof) * (2.0 * math.pi)

    # Sample C-space configurations (Jacobian-weighted + boundary-densified)
    q_samples, y_samples = None, None
    if args.sample:
        q_samples, y_samples = scene.sample_configurations(
            arm=arm,
            n_samples=5000,
            use_boundary_sampling=True,
            boundary_ratio=0.5,
            sigma=0.15,
        )

    visualizer = SceneVisualizer(scene, arm)
    visualizer.show(qs=qs, q_samples=q_samples, y_samples=y_samples)


if __name__ == "__main__":
    main()
