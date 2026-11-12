import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from metacosdf.torch_utility import torchify_as


@dataclass(frozen=True)
class Obstacle(ABC):
    @abstractmethod
    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): calculate sdf of given tensor p."""

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """Name for the obstacle type to be used in config files."""


@dataclass(frozen=True)
class Circle(Obstacle):
    center_x: float
    center_y: float
    radius: float

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        return (p - torchify((self.center_x, self.center_y))).norm(dim=-1) - self.radius


    @classmethod
    def config_key(cls) -> str:
        return "circle"


@dataclass(frozen=True)
class Box(Obstacle):
    center_x: float
    center_y: float
    half_size_x: float
    half_size_y: float
    angle: float = 0.0  # counter-clockwise rotation in radians

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        d = p - torchify((self.center_x, self.center_y))
        c, s = math.cos(self.angle), math.sin(self.angle)
        local = torch.stack(
            [c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], dim=-1
        )  # R^T d
        q = local.abs() - torchify((self.half_size_x, self.half_size_y))
        outside = q.clamp(min=0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0)
        return outside + inside

    @classmethod
    def config_key(cls) -> str:
        return "box"


@dataclass(frozen=True)
class Composition(Obstacle):
    elements: list[Obstacle]

    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): get the minimum of obstacle's SDFs."""
        d = self.elements[0].sdf(p)
        for obstacle in self.elements[1:]:
            d = torch.minimum(d, obstacle.sdf(p))
        return d

    @classmethod
    def config_key(cls) -> str:
        return "composition"
