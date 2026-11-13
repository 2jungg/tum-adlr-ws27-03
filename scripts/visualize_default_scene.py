from pathlib import Path

import matplotlib.pyplot as plt
import torch

from metacosdf.config_utils import load_config
from metacosdf.scene_generator.scene import cspace_grid
from metacosdf.scene_generator.visualizer import SceneVisualizer

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "default_scene.yaml"
    config_dict = load_config(config_path)
    visualizer = SceneVisualizer.from_dict(config_dict["scene_visualizer"])

    # (4, 2) hand-picked configurations in radians:
    # the first one collides with the circle, the other three are free
    poses = torch.tensor([[0.2, 0.2], [2.2, 0.4], [2.6, 0.9], [4.2, 0.8]])

    # workspace: obstacles, poses and reach | workspace SDF
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    visualizer.plot_workspace(
        axes[0], qs=poses, title="workspace, poses and the arm's reach (dashed)"
    )
    visualizer.plot_workspace(
        axes[1], qs=poses, show_sdf=True, title="workspace SDF, iso-lines every 0.5"
    )
    fig.tight_layout()

    # C-space: the ground-truth SDF and the two fields it is built from.
    # All three share the same zero level set, so every solid black line must lie
    # on the dashed ground-truth boundary.
    grid = cspace_grid(256)  # (256, 256, 2)
    ground_truth = visualizer.scene.cspace_sdf(256)  # (256, 256), radians
    visualizer.plot_overview(
        ground_truth,
        qs=poses,
        ncols=2,
        fields={
            "collision labels y(q)": visualizer.scene.collision_labels(grid),
            "clearance c(q) [workspace units]": visualizer.scene.clearance(grid),
        },
    )
    plt.show()
