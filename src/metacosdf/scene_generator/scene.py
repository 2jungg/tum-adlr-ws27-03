import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from metacosdf.config_utils import Config
from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.obstacles import Obstacle


def cspace_grid(resolution: int = 256) -> Tensor:
    """
    (res, res, 2) grid of configurations,
    "ij" indexing: grid[i, j] = (q1_i, q2_j), q_k = k * 2pi / res.
    """
    axis = torch.arange(resolution, dtype=torch.float32) * (2 * math.pi / resolution)
    q1, q2 = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([q1, q2], dim=-1)


@dataclass
class Scene(Config):
    obstacle: Obstacle
    robot: PlanarArm

    @dataclass
    class Bounds:
        xmin: float = -3.0
        xmax: float = 3.0
        ymin: float = -3.0
        ymax: float = 3.0

    bounds: Bounds = field(default_factory=Bounds)

    def workspace_grid(self, resolution: int = 300) -> Tensor:
        """
        (res, res, 2) grid of points over the scene bounds,
        "ij" indexing: grid[i, j] = (x_i, y_j).
        """
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(self.bounds.xmin, self.bounds.xmax, resolution),
            torch.linspace(self.bounds.ymin, self.bounds.ymax, resolution),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y], dim=-1)

    def clearance(self, q: Tensor, points_per_link: int = 32) -> Tensor:
        """(..., n) -> (...): return float clearance"""
        points = self.robot.link_points(
            q, points_per_link=points_per_link
        )  # (..., n * points_per_link, 2)
        return self.obstacle.sdf(points).amin(dim=-1) - self.robot.link_radius  # (...)

    def collision_labels(self, q: Tensor) -> Tensor:
        """(..., n) -> (...): return binary collision labels(-1 = collision, +1 = free)"""
        return torch.where(self.clearance(q) > 0, 1.0, -1.0)

    def cspace_sdf(self, resolution: int = 256) -> Tensor:
        """
        (res, res) ground-truth C-space SDF on cspace_grid(res),
        positive in free space.
        """
        labels = self.collision_labels(cspace_grid(resolution))
        free = (labels > 0).numpy()
        # without both classes, distance_transform_edt returns meaningless numbers
        # instead of failing
        if free.all() or not free.any():
            raise ValueError(
                f"C-space SDF needs both free and colliding configurations, but this scene is "
                f"{'entirely free' if free.all() else 'entirely in collision'}"
            )

        cell_size = 2 * math.pi / resolution
        # the C-space is a torus:
        # tile 3x3 so that distances across the 0 / 2pi seam are right, then crop the center
        free_tiled = np.tile(free, (3, 3))
        distance_to_collision = distance_transform_edt(free_tiled, sampling=cell_size)
        distance_to_free = distance_transform_edt(~free_tiled, sampling=cell_size)
        # the EDT measures between cell centers:
        # shift by half a cell to put the boundary between the cells
        signed_distance = np.where(
            free_tiled,
            distance_to_collision - cell_size / 2,
            -(distance_to_free - cell_size / 2),
        )
        center = slice(resolution, 2 * resolution)
        return torch.from_numpy(signed_distance[center, center]).float()
