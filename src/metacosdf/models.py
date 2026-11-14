"""Networks on the embedding phi(q) from fields.py, never on raw joint angles."""

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from metacosdf.config_utils import Config

ACTIVATIONS = {"relu": nn.ReLU, "softplus": nn.Softplus, "silu": nn.SiLU}


@dataclass
class MLPConfig(Config):
    hidden_width: int = 128
    hidden_layers: int = 3
    # smooth activations suit the Eikonal term
    activation: Literal["relu", "softplus", "silu"] = "silu"


class MLP(nn.Module):
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
