"""Random scenes: circles and boxes placed uniformly (by area) within the arm's reach."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from metacosdf.config_utils import Config
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Box, Circle, Composition, Obstacle
from metacosdf.scene_generator.scene import Scene


@dataclass
class SceneGeneratorConfig(Config):
    robot: PlanarArm = field(default_factory=PlanarArm)
    bounds: Scene.Bounds = field(default_factory=Scene.Bounds)
    max_obstacles: int = 4
    # obstacles keep at least this distance from the base, on top of the link radius
    min_base_clearance: float = 0.2


def create_random_obstacle(config: SceneGeneratorConfig, rng: np.random.Generator) -> Obstacle:
    """
    A circle or a box, with its center uniformly (by area) in the ring
    between the base clearance and the arm's reach
    """
    robot = config.robot
    angle = float(rng.uniform(0, 2 * math.pi))
    min_distance = config.min_base_clearance + robot.link_radius
    distance = math.sqrt(float(rng.uniform(min_distance**2, robot.reach**2)))
    center_x = robot.base_position_x + distance * math.cos(angle)
    center_y = robot.base_position_y + distance * math.sin(angle)
    # plain Python floats, so that scenes can be saved to YAML and to weights-only .pt files
    if rng.uniform() > 0.5:
        return Circle(center_x=center_x, center_y=center_y, radius=float(rng.uniform(0.1, 0.5)))
    return Box(
        center_x=center_x,
        center_y=center_y,
        half_size_x=float(rng.uniform(0.1, 0.5)),
        half_size_y=float(rng.uniform(0.1, 0.5)),
        angle=float(rng.uniform(0, math.pi)),
    )


def create_random_scene(config: SceneGeneratorConfig, rng: np.random.Generator) -> Scene:
    """1 to max_obstacles random obstacles; candidates too close to the base are rejected."""
    robot = config.robot
    n_obstacles = int(rng.integers(1, config.max_obstacles + 1))
    base = torch.tensor([robot.base_position_x, robot.base_position_y])
    obstacles: list[Obstacle] = []
    for _ in range(100):  # at most 100 candidates
        obstacle = create_random_obstacle(config, rng)
        if obstacle.sdf(base) > robot.link_radius + config.min_base_clearance:
            obstacles.append(obstacle)
            if len(obstacles) == n_obstacles:
                break
    return Scene(obstacle=Composition(elements=obstacles), robot=robot, bounds=config.bounds)
