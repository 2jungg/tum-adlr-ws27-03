import torch
from torch import Tensor
from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from metacosdf.scene_generator.obstacles import Obstacle
from metacosdf.scene_generator.arm import PlanarArm

def cspace_grid(n: int = 256) -> Tensor:
    axis = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    q1, q2 = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([q1, q2], dim=-1)

@dataclass(frozen=True)
class Scene:
    obstacles: tuple[Obstacle, ...]
    bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0)  # xmin, xmax, ymin, ymax

    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): get the minimum of obstacle's SDFs"""
        d = self.obstacles[0].sdf(p)
        for obstacle in self.obstacles[1:]:
            d = torch.minimum(d, obstacle.sdf(p))
        return d

    def clearance(self, arm: PlanarArm, q: Tensor, per_link: int = 32) -> Tensor:
        """(..., n) -> (...): return float clearance"""
        lp = arm.link_points(q, per_link=per_link)  # shape: (..., n * per_link, 2)
        return self.sdf(lp).amin(dim=-1) - arm.link_radius  # shape: (...,)

    def collision_labels(self, arm: PlanarArm, q: Tensor) -> Tensor:
        """(..., n) -> (...): return binary collision labels(-1 = collision, +1 = free)"""
        return torch.where(self.clearance(arm, q) > 0, 1.0, -1.0)

    def cspace_sdf(self, arm: PlanarArm, n: int = 256) -> Tensor:
        """return n x n ground truth cspace sdf"""
        grid = cspace_grid(n)
        gt_col = self.collision_labels(arm, grid)
        free = (gt_col > 0).numpy()

        unit = 2 * math.pi / n
        tiled = np.tile(free, (3, 3))

        to_col = distance_transform_edt(tiled, sampling=unit)
        to_free = distance_transform_edt(~tiled, sampling=unit)
        gt_dis = np.where(tiled, to_col - unit / 2, -(to_free - unit / 2))
        gt_sdf = torch.from_numpy(gt_dis[n : 2 * n, n : 2 * n]).float()
        return gt_sdf