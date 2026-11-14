from pathlib import Path

from metacosdf.config_utils import save_config
from metacosdf.dataset import DatasetConfig
from metacosdf.models import MLPConfig
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Circle
from metacosdf.scene_generator.scene import Scene
from metacosdf.scene_generator.visualizer import SceneVisualizer
from metacosdf.training import TrainingConfig

if __name__ == "__main__":
    configs_dir = Path(__file__).parent.parent / "configs"

    default_arm = PlanarArm()
    default_circle = Circle(center_x=1.2, center_y=0.5, radius=0.35)
    default_scene = Scene(obstacle=default_circle, robot=default_arm)
    default_visualizer = SceneVisualizer(scene=default_scene)
    config_dict = {
        "scene_visualizer": default_visualizer.to_dict(),
    }
    save_config(config_dict, configs_dir / "default_scene.yaml")

    save_config({"dataset": DatasetConfig().to_dict()}, configs_dir / "dataset.yaml")

    save_config(
        {"model": MLPConfig().to_dict(), "training": TrainingConfig().to_dict()},
        configs_dir / "fit_mlp.yaml",
    )
