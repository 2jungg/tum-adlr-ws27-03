import matplotlib.pyplot as plt
import torch

from metacosdf.robot import PlanarArm
from metacosdf.obstacles import Circle
from metacosdf.scene import Scene
from metacosdf.visualization import plot_workspace

if __name__ == "__main__":
    arm = PlanarArm()
    scene = Scene(obstacle=Circle(center=(3.2, 6.4), radius=0.6))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    qs = (0.3, 0.8)
    plot_workspace(axes[0], scene, arm, qs=torch.tensor(qs), labels=[f"q = {qs}"])
    axes[0].set_title("workspace, arm and its reach (dashed)")
    plot_workspace(axes[1], scene, show_sdf=True)
    axes[1].set_title("workspace SDF, iso-lines every 0.5")
    fig.tight_layout()
    plt.show()