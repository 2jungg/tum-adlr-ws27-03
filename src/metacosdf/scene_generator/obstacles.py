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


@dataclass
class Circle(Obstacle):
    center: tuple[float, float]
    radius: float

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        return torch.linalg.vector_norm(p - torchify(self.center), dim=-1) - self.radius


@dataclass
class Box(Obstacle):
    center: tuple[float, float]
    half_size: tuple[float, float]
    angle: float = 0.0  # counter-clockwise rotation in radians

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        offset = p - torchify(self.center)
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        local = torch.stack(
            [
                cos_angle * offset[..., 0] + sin_angle * offset[..., 1],
                -sin_angle * offset[..., 0] + cos_angle * offset[..., 1],
            ],
            dim=-1,
        )  # R^T offset: p in the box frame
        # by symmetry only one quadrant matters: the position relative to the box corner
        corner_offset = local.abs() - torchify(self.half_size)
        outside = torch.linalg.vector_norm(corner_offset.clamp(min=0), dim=-1)
        inside = corner_offset.amax(dim=-1).clamp(max=0)
        return outside + inside


@dataclass
class Arc(Obstacle):
    """Circular arc (horseshoe) with rounded ends. Inigo Quilez's sdArc."""

    center: tuple[float, float]
    radius: float
    thickness: float
    aperture: float  # half-angle in (0, pi); total angular span is 2 * aperture
    angle: float = 0.0  # counter-clockwise orientation in radians

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        offset = p - torchify(self.center)
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        # R^T offset: arc symmetry axis aligned with +y, opening facing -y
        local_x = (cos_angle * offset[..., 0] + sin_angle * offset[..., 1]).abs()
        local_y = -sin_angle * offset[..., 0] + cos_angle * offset[..., 1]
        local = torch.stack([local_x, local_y], dim=-1)

        sin_aperture, cos_aperture = math.sin(self.aperture), math.cos(self.aperture)
        outside_arc = cos_aperture * local_x > sin_aperture * local_y
        endpoint = torchify((self.radius * sin_aperture, self.radius * cos_aperture))
        dist_endpoint = torch.linalg.vector_norm(local - endpoint, dim=-1)
        dist_arc = (torch.linalg.vector_norm(local, dim=-1) - self.radius).abs()
        return torch.where(outside_arc, dist_endpoint, dist_arc) - self.thickness


@dataclass
class Polygon(Obstacle):
    """Simple polygon (convex or non-convex), vertices in order. Inigo Quilez's sdPolygon."""

    vertices: tuple[tuple[float, float], ...]

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        v = torchify(self.vertices)  # (V, 2)
        d2 = ((p - v[0]) ** 2).sum(dim=-1)
        sign = torch.ones_like(d2)
        for i in range(len(v)):
            j = i - 1
            edge = v[j] - v[i]
            w = p - v[i]
            t = ((w * edge).sum(dim=-1) / (edge * edge).sum()).clamp(0.0, 1.0)
            closest = w - t[..., None] * edge
            d2 = torch.minimum(d2, (closest * closest).sum(dim=-1))
            c1 = p[..., 1] >= v[i, 1]
            c2 = p[..., 1] < v[j, 1]
            c3 = edge[0] * w[..., 1] > edge[1] * w[..., 0]
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign = torch.where(flip, -sign, sign)
        return sign * d2.clamp(min=0.0).sqrt()


@dataclass
class Composition(Obstacle):
    elements: tuple[Obstacle, ...]

    def sdf(self, p: Tensor) -> Tensor:
        """(..., 2) -> (...): get the minimum of obstacle's SDFs."""
        distance = self.elements[0].sdf(p)
        for obstacle in self.elements[1:]:
            distance = torch.minimum(distance, obstacle.sdf(p))
        return distance
