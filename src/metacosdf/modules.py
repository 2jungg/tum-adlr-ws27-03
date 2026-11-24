"""Networks on the embedding phi(q) from fields.py, never on raw joint angles."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from metacosdf.config_utils import Config

ActivationFunctionName = Literal["relu", "softplus", "silu"]
ACTIVATIONS_TO_TORCH: dict[ActivationFunctionName, type[nn.Module]] = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


@dataclass
class ModuleConfig(Config, ABC):
    @abstractmethod
    def create_module(self, input_dim: int) -> nn.Module:
        pass


@dataclass
class MLPConfig(ModuleConfig):
    hidden_layer_width: int = 128
    hidden_layer_count: int = 3
    # smooth activations suit the Eikonal term
    activation_function: ActivationFunctionName = "silu"

    def create_module(self, input_dim: int) -> nn.Module:
        return MLP(
            input_dim=input_dim,
            hidden_layer_width=self.hidden_layer_width,
            hidden_layer_count=self.hidden_layer_count,
            activation_function=ACTIVATIONS_TO_TORCH[self.activation_function],
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_count: int,
        hidden_layer_width: int,
        activation_function: type[nn.Module],
    ) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_width))
        layers.append(activation_function())
        for _ in range(hidden_layer_count - 1):
            layers += [
                nn.Linear(hidden_layer_width, hidden_layer_width),
                activation_function(),
            ]
        layers.append(nn.Linear(hidden_layer_width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """(..., input_dim) -> (...)"""
        return self.layers(inputs).squeeze(-1)


def safe_inverse(x: Tensor) -> Tensor:
    """1 / x if x != 0, 0 if x == 0"""
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class LipschitzLayer(nn.Module):
    def __init__(self, width: int, activation_function: type[nn.Module]) -> None:
        super().__init__()
        self.activation_function = activation_function()
        self.weight = nn.Parameter(torch.empty(width, width))
        self.bias = nn.Parameter(torch.empty(width))
        self.q = nn.Parameter(torch.randn(width))
        nn.init.xavier_normal_(self.weight)
        bound = 1 / math.sqrt(width)
        nn.init.uniform_(self.bias, -bound, bound)

    def compute_t(self) -> Tensor:
        gram = self.weight.T @ self.weight
        q_ratio = torch.exp(self.q[None, :] - self.q[:, None])
        return (gram * q_ratio).abs().sum(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.activation_function(x @ self.weight + self.bias)
        return x - 2 * safe_inverse(self.compute_t()) * hidden @ self.weight.T


class NormalizedLinear(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        """(..., width) -> (...)"""
        return x @ self.weight / torch.linalg.vector_norm(self.weight) + self.bias


class SLL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_count: int,
        hidden_layer_width: int,
        activation_function: type[nn.Module],
    ) -> None:
        super().__init__()
        layers = []
        layers.append(nn.ZeroPad1d((0, hidden_layer_width - input_dim)))
        for _ in range(hidden_layer_count):
            layers.append(LipschitzLayer(hidden_layer_width, activation_function))
        layers.append(NormalizedLinear(hidden_layer_width))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """(..., input_dim) -> (...)"""
        return self.layers(inputs)


@dataclass
class SLLConfig(ModuleConfig):
    # Co-SDF, 2-DoF: 10 layers of width 256
    hidden_layer_width: int = 256
    hidden_layer_count: int = 10
    activation_function: ActivationFunctionName = "relu"

    def create_module(self, input_dim: int) -> SLL:
        return SLL(
            input_dim,
            self.hidden_layer_count,
            self.hidden_layer_width,
            activation_function=ACTIVATIONS_TO_TORCH[self.activation_function],
        )
