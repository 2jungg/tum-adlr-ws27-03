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
class SamplingConfig(Config):
    """How labeled configurations are sampled in C-space (Co-SDF, Sec. III-B)."""

    n_samples: int = 5000
    # add samples close to the collision boundary
    boundary_sampling: bool = True
    # fraction of the samples taken close to the boundary
    boundary_ratio: float = 0.5
    # std of the Gaussian perturbation that makes the boundary pairs, in radians
    sigma: float = 0.15
    # bisection steps that move both ends of a boundary pair towards the boundary
    bisection_steps: int = 3


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

    def sample_configurations(self, config: SamplingConfig) -> tuple[Tensor, Tensor]:
        """
        Jacobian-weighted samples, plus samples close to the collision boundary if enabled
        -> shuffled q (N, n) in [0, 2pi) and their collision labels y (N,)
        """
        n_boundary = int(config.n_samples * config.boundary_ratio)
        if not config.boundary_sampling:
            n_boundary = 0
        q_boundary = torch.empty(0, self.robot.dof)
        if n_boundary > 0:
            # pairs: a weighted sample and a Gaussian perturbation of it
            q_start = self.robot.weighted_sample(max(5 * n_boundary, 1000))
            q_end = q_start + config.sigma * torch.randn_like(q_start)
            # keep the pairs whose labels differ: the segment between them crosses the boundary
            y_start = self.collision_labels(q_start)
            crossing = y_start != self.collision_labels(q_end)
            q_start, q_end, y_start = q_start[crossing], q_end[crossing], y_start[crossing]
            # bisection: both ends move towards the boundary and keep their labels
            for _ in range(config.bisection_steps):
                q_middle = 0.5 * (q_start + q_end)
                # (pairs, 1)
                middle_like_start = (self.collision_labels(q_middle) == y_start)[:, None]
                q_start = torch.where(middle_like_start, q_middle, q_start)
                q_end = torch.where(middle_like_start, q_end, q_middle)
            # (2 * pairs, n): both ends of each pair next to each other, so that truncating
            # keeps the two sides of the boundary balanced
            q_boundary = torch.stack([q_start, q_end], dim=1).flatten(0, 1)[:n_boundary]

        q_weighted = self.robot.weighted_sample(config.n_samples - len(q_boundary))
        q = torch.cat([q_weighted, q_boundary])
        q = torch.remainder(q[torch.randperm(len(q))], 2 * math.pi)
        return q, self.collision_labels(q)
