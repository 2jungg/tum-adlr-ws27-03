"""The default scene from configs/: workspace, workspace SDF and ground-truth C-space fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from metacosdf.scene_generator.scene import Scene, cspace_grid
from metacosdf.visualization import SceneVisualizationConfig, plot_overview, plot_workspace

if __name__ == "__main__":
    configs_dir = Path(__file__).parent.parent / "configs"
    scene = Scene.from_yaml(configs_dir / "default_scene.yaml")
    config = SceneVisualizationConfig.from_yaml(configs_dir / "scene_visualization.yaml")

    # (4, 2) hand-picked configurations in radians:
    # two in collision, one just free next to the box, one far away
    poses = torch.tensor([[0.2, 0.2], [2.2, 0.4], [2.6, 0.9], [4.2, 0.8]])

    # workspace: obstacles, poses and reach | workspace SDF
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_workspace(
        scene,
        ax=axes[0],
        qs=poses,
        title="workspace, poses and the arm's reach (dashed)",
        config=config,
    )
    plot_workspace(
        scene,
        ax=axes[1],
        qs=poses,
        show_sdf=True,
        title="workspace SDF, iso-lines every 0.5",
        config=config,
    )
    fig.tight_layout()

    # C-space: the ground-truth SDF and the two fields it is built from.
    # All three share the same zero level set, so every solid black line must lie
    # on the dashed ground-truth boundary.
    grid = cspace_grid(256)  # (256, 256, 2)
    ground_truth = scene.cspace_sdf(256)  # (256, 256), radians
    plot_overview(
        scene,
        ground_truth,
        qs=poses,
        ncols=2,
        fields={
            "collision labels y(q)": scene.collision_labels(grid),
            "clearance c(q) [workspace units]": scene.clearance(grid),
        },
        config=config,
    )
    plt.show()
