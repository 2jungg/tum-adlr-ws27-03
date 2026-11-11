import torch
from torch import Tensor
from dataclasses import dataclass
import math

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

    def cspace_sdf(self) -> Tensor:
        pass