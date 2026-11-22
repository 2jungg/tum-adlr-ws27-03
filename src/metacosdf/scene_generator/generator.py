"""Random scenes of circles, boxes, arcs and polygons within the arm's reach."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from metacosdf.config_utils import Config
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Arc, Box, Circle, Composition, Obstacle, Polygon
from metacosdf.scene_generator.scene import Scene


@dataclass
class SceneGeneratorConfig(Config):
    robot: PlanarArm = field(default_factory=PlanarArm)
    bounds: Scene.Bounds = field(default_factory=Scene.Bounds)
    max_obstacles: int = 4
    # distance from the base, on top of the link radius
    min_base_clearance: float = 0.2
    obstacle_types: tuple[str, ...] = ("circle", "box", "arc", "polygon")


def _create_random_polygon(center: tuple[float, float], rng: np.random.Generator) -> Polygon:
    """Star-shaped simple polygon (convex or non-convex) without self-intersections."""
    center_x, center_y = center
    n_vertices = int(rng.integers(5, 9))
    step = 2 * math.pi / n_vertices
    rotation = float(rng.uniform(0, 2 * math.pi))
    vertices: list[tuple[float, float]] = []
    for k in range(n_vertices):
        theta = k * step + float(rng.uniform(-0.35 * step, 0.35 * step)) + rotation
        radius = float(rng.uniform(0.12, 0.5))
        vertices.append(
            (center_x + radius * math.cos(theta), center_y + radius * math.sin(theta))
        )
    return Polygon(vertices=tuple(vertices))


def create_random_obstacle(config: SceneGeneratorConfig, rng: np.random.Generator) -> Obstacle:
    """random obstacle centered uniformly by area between the base clearance and the reach"""
    robot = config.robot
    angle = float(rng.uniform(0, 2 * math.pi))
    min_distance = config.min_base_clearance + robot.link_radius
    distance = math.sqrt(float(rng.uniform(min_distance**2, robot.reach**2)))
    base_x, base_y = robot.base_position
    center = (base_x + distance * math.cos(angle), base_y + distance * math.sin(angle))
    # python floats: numpy floats break YAML and weights-only torch.load
    obstacle_type = str(rng.choice(config.obstacle_types))
    if obstacle_type == "circle":
        return Circle(center=center, radius=float(rng.uniform(0.1, 0.5)))
    if obstacle_type == "box":
        return Box(
            center=center,
            half_size=(float(rng.uniform(0.1, 0.5)), float(rng.uniform(0.1, 0.5))),
            angle=float(rng.uniform(0, math.pi)),
        )
    if obstacle_type == "arc":
        return Arc(
            center=center,
            radius=float(rng.uniform(0.25, 0.55)),
            thickness=float(rng.uniform(0.06, 0.14)),
            aperture=float(rng.uniform(0.5 * math.pi, 0.85 * math.pi)),
            angle=float(rng.uniform(0, 2 * math.pi)),
        )
    if obstacle_type == "polygon":
        return _create_random_polygon(center, rng)
    raise ValueError(f"Unknown obstacle type: {obstacle_type!r}")


def create_random_scene(config: SceneGeneratorConfig, rng: np.random.Generator) -> Scene:
    """1 to max_obstacles obstacles, rejecting the ones too close to the base"""
    robot = config.robot
    n_obstacles = int(rng.integers(1, config.max_obstacles + 1))
    base = torch.tensor(robot.base_position)
    obstacles: list[Obstacle] = []
    for _ in range(100):
        obstacle = create_random_obstacle(config, rng)
        if obstacle.sdf(base) > robot.link_radius + config.min_base_clearance:
            obstacles.append(obstacle)
            if len(obstacles) == n_obstacles:
                break
    return Scene(
        obstacle=Composition(elements=tuple(obstacles)), robot=robot, bounds=config.bounds
    )
