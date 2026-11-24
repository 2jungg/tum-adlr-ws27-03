import math
from pathlib import Path

from metacosdf.config_utils import save_config
from metacosdf.dataset import DatasetConfig
from metacosdf.losses import HKRConfig, LossConfig, MSEConfig
from metacosdf.modules import MLPConfig, ModuleConfig, SIRENConfig, SLLConfig
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Arc, Box, Circle, Composition, Polygon
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

    # all four obstacle types, 58% of the C-space in collision. The arm can reach into the
    # horseshoe and into the bracket's slot -> narrow corridors and isolated free islands
    cluttered_obstacles = Composition(
        elements=(
            # a horseshoe whose opening faces the base
            Arc(
                center=(-0.9, 0.9), radius=0.55, thickness=0.08, aperture=2.2, angle=math.pi / 4
            ),
            # a U-shaped bracket, its slot open towards the base
            Polygon(
                vertices=(
                    (1.0, 0.35),
                    (1.75, 0.35),
                    (1.75, -0.75),
                    (1.0, -0.75),
                    (1.0, -0.47),
                    (1.5, -0.47),
                    (1.5, 0.07),
                    (1.0, 0.07),
                )
            ),
            # 0.56 from the base: link 1 can hit it -> a full-height vertical band in C-space
            Circle(center=(-0.55, -0.5), radius=0.18),
            # at the edge of the reach: only the tip of link 2 gets there
            Box(center=(0.25, -1.65), half_size=(0.28, 0.12), angle=0.4),
        )
    )
    cluttered_scene = Scene(obstacle=cluttered_obstacles, robot=default_arm)
    cluttered_scene.to_yaml(configs_dir / "cluttered_scene.yaml")

    SceneVisualizationConfig().to_yaml(configs_dir / "scene_visualization.yaml")
    DatasetConfig().to_yaml(configs_dir / "dataset.yaml")

    fits = {
        "fit_mlp_no_eikonal.yaml": (
            MLPConfig(),
            MSEConfig(eikonal_weight=0.0),
            TrainingConfig(),
        ),
        "fit_mlp.yaml": (MLPConfig(), MSEConfig(), TrainingConfig()),
        # MSE + Eikonal as in Co-SDF; lr 1e-4 as in SIREN's code (see SIRENConfig.omega)
        "fit_siren.yaml": (SIRENConfig(), MSEConfig(), TrainingConfig(learning_rate=1e-4)),
        "fit_cosdf.yaml": (SLLConfig(), HKRConfig(), TrainingConfig()),
    }
    for file_name, (module_config, loss_config, training_config) in fits.items():
        save_config(
            {
                ModuleConfig.key(): module_config.to_dict(),
                LossConfig.key(): loss_config.to_dict(),
                TrainingConfig.key(): training_config.to_dict(),
            },
            configs_dir / file_name,
        )
