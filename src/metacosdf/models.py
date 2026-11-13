"""Networks g: (..., input_dim) -> (...).

A network gets the already transformed configuration phi(q), never raw joint angles:
the transformation is shared by all networks and lives in fields.py.
"""

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from metacosdf.config_utils import Config

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


@dataclass
class MLPConfig(Config):
    hidden_width: int = 128
    hidden_layers: int = 3
    # smooth activations (softplus, silu) give the Eikonal term useful second derivatives
    activation: Literal["relu", "softplus", "silu"] = "silu"


class MLP(nn.Module):
    """A plain fully connected network: input_dim -> hidden_layers x hidden_width -> 1."""

    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(width, config.hidden_width), ACTIVATIONS[config.activation]()]
            width = config.hidden_width
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """(..., input_dim) -> (...)"""
        return self.layers(inputs).squeeze(-1)
