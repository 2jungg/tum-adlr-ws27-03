from typing import Protocol
from dataclasses import dataclass

from torch import Tensor

from metacosdf.torch_utility import torchify_as


class Obstacle(Protocol):
    def sdf(self, p: Tensor) -> Tensor:
        pass


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float

    def sdf(self, p: Tensor) -> Tensor:
        torchify = torchify_as(p)
        return (p - torchify(self.center)).norm(dim=-1) - self.radius
