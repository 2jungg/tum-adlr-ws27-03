import torch
from torch import Tensor
from dataclasses import dataclass

from math import pi

@dataclass(frozen=True)
class PlanarArm:
    link_lengths: tuple[float, ...] = (1.0, 1.0)
    base: tuple[float, float] = (0.0, 0.0)
    link_radius: float = 0.1

    @property
    def dof(self) -> int:
        return len(self.link_lengths)

    @property
    def reach(self) -> float:
        return sum(self.link_lengths)

    def joint_positions(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., n+1, 2): base, joint 2, ... end effector."""
        angles = torch.cumsum(q, dim=-1)  # (..., n) absolute link angles
        lengths = torch.as_tensor(self.link_lengths, dtype=q.dtype, device=q.device)
        link_vecs = lengths[:, None] * torch.stack([angles.cos(), angles.sin()], dim=-1)  # (..., n, 2)
        base = torch.as_tensor(self.base, dtype=q.dtype, device=q.device)
        joints = base + torch.cumsum(link_vecs, dim=-2)  # (..., n, 2)
        return torch.cat([base.expand(*q.shape[:-1], 1, 2), joints], dim=-2)

    def control_points(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., n, 2): the link centers."""
        p = self.joint_positions(q)
        return 0.5 * (p[..., :-1, :] + p[..., 1:, :])

    def embedding(self, q: Tensor) -> Tensor:
        """(..., n) -> (..., 2n): flattened control points formatted as network input"""
        return self.control_points(q).flatten(-2)

    def link_points(self, q: Tensor, per_link: int = 32) -> Tensor:
        """(..., n) -> (..., n * per_link, 2): points along every link, which needs for collision checking."""
        p = self.joint_positions(q)
        start, end = p[..., :-1, None, :], p[..., 1:, None, :]  # (..., n, 1, 2)
        s = torch.linspace(0, 1, per_link, dtype=q.dtype, device=q.device)[:, None]  # (per_link, 1)
        return (start + s * (end - start)).flatten(-3, -2)  # (..., n, per_link, 2) -> (..., n*per_link, 2)

if __name__ == "__main__":
    test_arm = PlanarArm()
    print(test_arm.joint_positions(Tensor((0, 0))))
    print(test_arm.joint_positions(Tensor((pi/2, 0))))
    print(test_arm.joint_positions(Tensor((0, pi/2))))
