import matplotlib.pyplot as plt
import torch

from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Circle, Composition, Box
from metacosdf.scene_generator.scene import Scene
from metacosdf.scene_generator.visualizer import SceneVisualizer

if __name__ == "__main__":
    arm = PlanarArm()

    # First Scene
    scene = Scene(obstacle=Circle(center_x=3.2, center_y=6.4, radius=0.6), robot=arm)
    scene_visualizer = SceneVisualizer(scene=scene)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    qs = (0.3, 0.8)
    scene_visualizer.plot_workspace(axes[0], qs=torch.tensor(qs), labels=[f"q = {qs}"])
    axes[0].set_title("workspace, arm and its reach (dashed)")
    scene_visualizer.plot_workspace(axes[1], show_sdf=True)
    axes[1].set_title("workspace SDF, iso-lines every 0.5")
    fig.tight_layout()
    plt.show()

    # Second Scene
    scene = Scene(
        obstacle=Composition(
            elements=[
                Circle(center_x=1.2, center_y=0.5, radius=0.35),
                Box(center_x=-0.8, center_y=0.9, half_size_x=0.3, half_size_y=0.35, angle=0.5),
            ]
        ),
        robot=arm,
    )
    qs = (
        (0.2, 0.2),
        (2.2, 0.4),
        (4.2, 0.8),
    )
    SceneVisualizer(scene=scene).show(qs=torch.tensor(qs), show_ws_sdf=False)
