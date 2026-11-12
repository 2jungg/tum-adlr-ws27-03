from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

import numpy as np
from scipy.ndimage import distance_transform_edt

from metacosdf.scene_generator.obstacles import Obstacle
from metacosdf.robot import PlanarArm


def cspace_grid(n: int = 256) -> Tensor:
    axis = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    q1, q2 = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([q1, q2], dim=-1)


@dataclass(frozen=True)
class Scene:
    obstacle: Obstacle
    robot: PlanarArm

    @dataclass(frozen=True)
    class Bounds:
        xmin: float = -3.0
        xmax: float = 3.0
        ymin: float = -3.0
        ymax: float = 3.0

    bounds: Bounds = field(default_factory=Bounds)

    def workspace_grid(self, resolution: int = 300) -> tuple[Tensor, ...]:
        """xy grid over the scene bounds: X and Y, each (res, res), indexing="ij"."""
        return torch.meshgrid(
            torch.linspace(self.bounds.xmin, self.bounds.xmax, resolution),
            torch.linspace(self.bounds.ymin, self.bounds.ymax, resolution),
            indexing="ij",
        )

    def clearance(self, q: Tensor, points_per_link: int = 32) -> Tensor:
        """(..., n) -> (...): return float clearance"""
        lp = self.robot.link_points(q, points_per_link=points_per_link)  # shape: (..., n * per_link, 2)
        return self.obstacle.sdf(lp).amin(dim=-1) - self.robot.link_radius  # shape: (...,)

    def collision_labels(self, q: Tensor) -> Tensor:
        """(..., n) -> (...): return binary collision labels(-1 = collision, +1 = free)"""
        return torch.where(self.clearance(q) > 0, 1.0, -1.0)

    def cspace_sdf(self, n: int = 256) -> Tensor:
        """return n x n ground truth cspace sdf"""
        grid = cspace_grid(n)
        gt_col = self.collision_labels(grid)
        free = (gt_col > 0).numpy()
        # without both classes, distance_transform_edt returns meaningless numbers instead of failing
        if free.all() or not free.any():
            raise ValueError(
                "C-space SDF needs both free and colliding configurations, "
                f"but this scene is {'entirely free' if free.all() else 'entirely in collision'}"
            )

        unit = 2 * math.pi / n
        tiled = np.tile(free, (3, 3))

        to_col = distance_transform_edt(tiled, sampling=unit)
        to_free = distance_transform_edt(~tiled, sampling=unit)
        gt_dis = np.where(tiled, to_col - unit / 2, -(to_free - unit / 2))
        gt_sdf = torch.from_numpy(gt_dis[n : 2 * n, n : 2 * n]).float()
        return gt_sdf
