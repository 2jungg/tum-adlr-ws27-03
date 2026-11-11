import torch
from torch import Tensor
from dataclasses import dataclass

import math
from abc import ABC, abstractmethod

def _t(x, like: Tensor) -> Tensor:
    """Convert python number/tuple to Tensor which have same dtype, device as input Tensor like"""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)

class Obstacle(ABC):
    @abstractmethod
    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): calculate sdf of given tensor p"""

    @abstractmethod
    def to_dict(self) -> dict:
        """save metadata of obstacle"""

@dataclass(frozen=True)
class Circle(Obstacle):
    center: tuple[float, float]
    radius: float

    def sdf(self, p: Tensor) -> Tensor:
        return (p - _t(self.center, p)).norm(dim=-1) - self.radius

    def to_dict(self) -> dict:
        return {
            "type": "circle",
            "center": [float(x) for x in self.center],
            "radius": float(self.radius),
        }


@dataclass(frozen=True)
class Box(Obstacle):
    center: tuple[float, float]
    half_size: tuple[float, float]
    angle: float = 0.0  # counter-clockwise rotation in radians

    def sdf(self, p: Tensor) -> Tensor:
        d = p - _t(self.center, p)
        c, s = math.cos(self.angle), math.sin(self.angle)
        local = torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], dim=-1)  # R^T d
        q = local.abs() - _t(self.half_size, p)
        outside = q.clamp(min=0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0)
        return outside + inside

    def to_dict(self) -> dict:
        return {
            "type": "box",
            "center": [float(x) for x in self.center],
            "half_size": [float(x) for x in self.half_size],
            "angle": float(self.angle),
        }