"""Example scene with a circle and a box: the workspace, its SDF, and the ground-truth C-space fields."""

import matplotlib.pyplot as plt
import torch

from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Box, Circle, Composition
from metacosdf.scene_generator.scene import Scene, cspace_grid
from metacosdf.scene_generator.visualizer import SceneVisualizer

if __name__ == "__main__":
    # arm with links (1, 1) at the origin: link 1 reaches up to 1.1 from the base, the whole arm up to 2.1
    scene = Scene(
        obstacle=Composition(
            elements=[
                # 0.95 from the base: link 1 can hit it -> a full-height vertical band in C-space
                Circle(center_x=1.2, center_y=0.5, radius=0.35),
                # 1.16 from the base: only link 2 can reach it -> a curved blob in C-space
                Box(center_x=-1.2, center_y=1.1, half_size_x=0.3, half_size_y=0.4, angle=0.5),
            ]
        ),
        robot=PlanarArm(),
    )
    visualizer = SceneVisualizer(scene=scene)
    # (4, 2) hand-picked configurations in radians: two in collision, one just free next to the box, one far away
    poses = torch.tensor([[0.2, 0.2], [2.2, 0.4], [2.6, 0.9], [4.2, 0.8]])

    # workspace: obstacles, poses and reach | workspace SDF
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    visualizer.plot_workspace(axes[0], qs=poses, title="workspace, poses and the arm's reach (dashed)")
    visualizer.plot_workspace(axes[1], qs=poses, show_sdf=True, title="workspace SDF, iso-lines every 0.5")
    fig.tight_layout()

    # C-space: the ground-truth SDF and the two fields it is built from. All three share the same zero
    # level set, so every solid black line must lie on the dashed ground-truth boundary.
    grid = cspace_grid(256)  # (256, 256, 2)
    ground_truth = scene.cspace_sdf(256)  # (256, 256), radians
    visualizer.plot_overview(
        ground_truth,
        qs=poses,
        ncols=2,
        fields={
            "collision labels y(q)": scene.collision_labels(grid),
            "clearance c(q) [workspace units]": scene.clearance(grid),
        },
    )
    plt.show()
