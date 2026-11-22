from pathlib import Path

from metacosdf.config_utils import save_config
from metacosdf.dataset import DatasetConfig
from metacosdf.models import MLPConfig
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Box, Circle, Composition
from metacosdf.scene_generator.scene import Scene
from metacosdf.training import TrainingConfig
from metacosdf.visualization import SceneVisualizationConfig

if __name__ == "__main__":
    configs_dir = Path(__file__).parent.parent / "configs"

    default_arm = PlanarArm()
    default_circle = Composition(
        elements=(
            # 0.95 from the base: link 1 can hit it
            # -> a full-height vertical band in C-space
            Circle(center=(1.2, 0.5), radius=0.35),
            # 1.16 from the base: only link 2 can reach it
            # -> a curved blob in C-space
            Box(center=(-1.2, 1.1), half_size=(0.3, 0.4), angle=0.5),
        )
    )
    default_scene = Scene(obstacle=default_circle, robot=default_arm)
    save_config({"scene": default_scene.to_dict()}, configs_dir / "default_scene.yaml")

    save_config(
        {"scene_visualization": SceneVisualizationConfig().to_dict()},
        configs_dir / "scene_visualization.yaml",
    )

    save_config({"dataset": DatasetConfig().to_dict()}, configs_dir / "dataset.yaml")

    save_config(
        {"model": MLPConfig().to_dict(), "training": TrainingConfig().to_dict()},
        configs_dir / "fit_mlp.yaml",
    )
