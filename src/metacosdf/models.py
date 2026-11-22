"""Networks on the embedding phi(q) from fields.py, never on raw joint angles."""

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from metacosdf.config_utils import Config

ActivationName = Literal["relu", "softplus", "silu"]
ACTIVATIONS_TO_TORCH: dict[ActivationName, type[nn.Module]] = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


@dataclass
class ModelConfig(Config):
    """Base of the model configs."""


@dataclass
class MLPConfig(ModelConfig):
    hidden_width: int = 128
    hidden_layers: int = 3
    # smooth activations suit the Eikonal term
    activation: ActivationName = "silu"


class MLP(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(config.hidden_layers):
            layers += [
                nn.Linear(width, config.hidden_width),
                ACTIVATIONS_TO_TORCH[config.activation](),
            ]
            width = config.hidden_width
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """(..., input_dim) -> (...)"""
        return self.layers(inputs).squeeze(-1)
