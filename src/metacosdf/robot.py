import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from metacosdf.config_utils import Config
from metacosdf.tensor_utils import torchify_as


@dataclass
class PlanarArm(Config):
    """Planar serial arm with revolute joints."""

    link_lengths: list[float] = field(default_factory=lambda: [1.0, 1.0])
    base_position_x: float = 0.0
    base_position_y: float = 0.0
    link_radius: float = 0.1

    @property
    def dof(self) -> int:
        return len(self.link_lengths)

    @property
    def reach(self) -> float:
        return sum(self.link_lengths)

    def joint_positions(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., n+1, 2)"""
        torchify = torchify_as(q)
        # (..., n)
        angles = torch.cumsum(q, dim=-1)
        # (n)
        lengths = torchify(self.link_lengths)
        # (..., n, 2)
        link_vectors = lengths[:, None] * torch.stack([angles.cos(), angles.sin()], dim=-1)
        # (2)
        base = torchify([self.base_position_x, self.base_position_y])
        # (..., n, 2)
        joints = base + torch.cumsum(link_vectors, dim=-2)
        return torch.cat([base.expand(*q.shape[:-1], 1, 2), joints], dim=-2)

    def control_points(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., n, 2)"""
        # (..., n+1, 2)
        p = self.joint_positions(q)
        return 0.5 * (p[..., :-1, :] + p[..., 1:, :])

    def embedding(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., 2n)"""
        return self.control_points(q).flatten(-2)

    def link_points(self, q: Tensor, points_per_link: int = 32) -> Tensor:
        """
        (..., n) -> (..., n * points_per_link, 2): evenly spaced points along every link,
        endpoints included
        """
        # (..., n+1, 2)
        p = self.joint_positions(q)
        # (..., n, 1, 2)
        start = p[..., :-1, None, :]
        # (..., n, 1, 2)
        end = p[..., 1:, None, :]
        # (points_per_link, 1): fractions 0 ... 1 of the way from the start to the end of a link
        fractions = torch.linspace(0, 1, points_per_link, dtype=q.dtype, device=q.device)[
            :, None
        ]
        # (..., n, points_per_link, 2)
        link_points = start + fractions * (end - start)
        return link_points.flatten(-3, -2)

    def jacobian_volume(self, q: Tensor) -> Tensor:
        """
        (B, n) -> (B,): sqrt(det(J^T J)) with J = d embedding / dq,
        the Co-SDF sampling weight
        """
        # (B, 2n, n)
        jacobian = torch.func.vmap(torch.func.jacfwd(self.embedding))(q)
        # (B, n, n)
        gram = jacobian.mT @ jacobian
        return torch.linalg.det(gram).clamp(min=0.0).sqrt()

    def weighted_sample(self, n_samples: int, oversample: int = 5) -> Tensor:
        """
        (n_samples, n) configurations in [0, 2pi), drawn from oversample * n_samples uniform
        candidates with probabilities proportional to jacobian_volume
        """
        # (oversample * n_samples, n)
        candidates = torch.rand(n_samples * oversample, self.dof) * (2 * math.pi)
        weights = self.jacobian_volume(candidates)
        return candidates[torch.multinomial(weights, n_samples, replacement=False)]
