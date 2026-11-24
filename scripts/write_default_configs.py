from pathlib import Path

from metacosdf.config_utils import save_config
from metacosdf.dataset import DatasetConfig
from metacosdf.losses import HKRConfig, LossConfig, MSEConfig
from metacosdf.modules import MLPConfig, ModuleConfig, SLLConfig
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
    default_scene.to_yaml(configs_dir / "default_scene.yaml")
    SceneVisualizationConfig().to_yaml(configs_dir / "scene_visualization.yaml")
    DatasetConfig().to_yaml(configs_dir / "dataset.yaml")

    # one file per fit: module, loss and training, the same training for every fit
    fits = {
        "fit_mlp_no_eikonal.yaml": (MLPConfig(), MSEConfig(eikonal_weight=0.0)),
        "fit_mlp.yaml": (MLPConfig(), MSEConfig()),
        "fit_cosdf.yaml": (SLLConfig(), HKRConfig()),
    }
    for file_name, (module_config, loss_config) in fits.items():
        save_config(
            {
                ModuleConfig.key(): module_config.to_dict(),
                LossConfig.key(): loss_config.to_dict(),
                TrainingConfig.key(): TrainingConfig().to_dict(),
            },
            configs_dir / file_name,
        )
