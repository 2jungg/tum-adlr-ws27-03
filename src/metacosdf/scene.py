from dataclasses import dataclass, field

import torch
from torch import Tensor

from metacosdf.obstacles import Obstacle


@dataclass(frozen=True)
class Scene:
    obstacle: Obstacle

    @dataclass(frozen=True)
    class Bounds:
        xmin: float = 0.0
        xmax: float = 1.0
        ymin: float = 0.0
        ymax: float = 1.0

    bounds: Bounds = field(default_factory=Bounds)
    field_color_map: str = "RdBu_r"
    obstacle_color: str = "#2166ac"

    def workspace_grid(self, resolution: int = 300) -> tuple[Tensor, ...]:
        """xy grid over the scene bounds: X and Y, each (res, res), indexing="ij"."""
        return torch.meshgrid(
            torch.linspace(self.bounds.xmin, self.bounds.xmax, resolution),
            torch.linspace(self.bounds.ymin, self.bounds.ymax, resolution),
            indexing="ij",
        )
