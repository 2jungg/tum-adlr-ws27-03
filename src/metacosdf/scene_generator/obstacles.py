import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from metacosdf.config_utils import Config
from metacosdf.tensor_utils import torchify_as


@dataclass
class Obstacle(Config, ABC):
    @abstractmethod
    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): calculate sdf of given tensor p."""

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """Name for the obstacle type to be used in config files."""


@dataclass
class Circle(Obstacle):
    center_x: float
    center_y: float
    radius: float

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        return (
            torch.linalg.vector_norm(p - torchify((self.center_x, self.center_y)), dim=-1)
            - self.radius
        )

    @classmethod
    def config_key(cls) -> str:
        return "circle"


@dataclass
class Box(Obstacle):
    center_x: float
    center_y: float
    half_size_x: float
    half_size_y: float
    angle: float = 0.0  # counter-clockwise rotation in radians

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        offset = p - torchify((self.center_x, self.center_y))
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        local = torch.stack(
            [
                cos_angle * offset[..., 0] + sin_angle * offset[..., 1],
                -sin_angle * offset[..., 0] + cos_angle * offset[..., 1],
            ],
            dim=-1,
        )  # R^T offset: p in the box frame
        # by symmetry only one quadrant matters: the position relative to the box corner
        corner_offset = local.abs() - torchify((self.half_size_x, self.half_size_y))
        outside = torch.linalg.vector_norm(corner_offset.clamp(min=0), dim=-1)
        inside = corner_offset.amax(dim=-1).clamp(max=0)
        return outside + inside

    @classmethod
    def config_key(cls) -> str:
        return "box"


@dataclass
class Composition(Obstacle):
    elements: list[Obstacle]

    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): get the minimum of obstacle's SDFs."""
        distance = self.elements[0].sdf(p)
        for obstacle in self.elements[1:]:
            distance = torch.minimum(distance, obstacle.sdf(p))
        return distance

    @classmethod
    def config_key(cls) -> str:
        return "composition"
