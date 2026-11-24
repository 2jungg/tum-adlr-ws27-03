"""Losses on collision labels y in {-1, +1} (-1: collision) for a field d(q)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from metacosdf.config_utils import Config


@dataclass
class LossConfig(Config, ABC):
    # weight of mean((|grad_q d| - 1)^2), Co-SDF Eq. (5); 0 skips the term
    eikonal_weight: float = 0.1

    @abstractmethod
    def loss_terms_values(self, distance: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """distance (B,), labels (B,) -> unweighted terms"""

    @abstractmethod
    def loss_terms_weights(self, epoch: int) -> dict[str, float]:
        """weights of the terms of loss_terms_values; epochs count from 1"""


@dataclass
class MSEConfig(LossConfig):
    """mean((d - y)^2), Co-SDF's loss for its MLP and SIREN baselines"""

    def loss_terms_values(self, distance: Tensor, labels: Tensor) -> dict[str, Tensor]:
        return {"mse": mean_squared_error(distance, labels)}

    def loss_terms_weights(self, epoch: int) -> dict[str, float]:
        return {"mse": 1.0}


@dataclass
class HKRConfig(LossConfig):
    """
    hKR, Co-SDF Eq. (2)-(4): mean(-y d) + lambda * mean(max(0, m - y d)).
    The KR mean runs over the whole batch, so each class counts with its share of the
    samples; Coiffier & Bethune's code takes one mean per class instead.
    """

    margin: float = 1e-2
    # (first epoch, lambda), sorted by epoch. Co-SDF only says "from 1 to 100"; these steps
    # are from Coiffier & Bethune's train_lip.py
    hinge_weight_steps: tuple[tuple[int, float], ...] = ((1, 1.0), (5, 10.0), (10, 100.0))

    def loss_terms_values(self, distance: Tensor, labels: Tensor) -> dict[str, Tensor]:
        # y d > 0 where the sign is right
        signed = labels * distance
        return {"kr": kr_term(signed), "hinge": hinge_term(signed, self.margin)}

    def loss_terms_weights(self, epoch: int) -> dict[str, float]:
        hinge_weight = self.hinge_weight_steps[0][1]
        for first_epoch, weight in self.hinge_weight_steps:
            if first_epoch <= epoch:
                hinge_weight = weight
        return {"kr": 1.0, "hinge": hinge_weight}


def field_loss(
    field: nn.Module, q: Tensor, labels: Tensor, config: LossConfig, epoch: int
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    q (B, n), labels (B,) -> (weighted total, unweighted terms detached for logging).
    The Eikonal gradient is taken w.r.t. q as in Co-SDF Eq. (5), although the SLL network
    is 1-Lipschitz in phi(q), not in q.
    """
    if config.eikonal_weight > 0:
        q = q.detach().requires_grad_(True)
    distance = field(q)
    terms = config.loss_terms_values(distance, labels)
    weights = config.loss_terms_weights(epoch)
    if config.eikonal_weight > 0:
        # create_graph: the Eikonal term is differentiated too
        (gradient,) = torch.autograd.grad(distance.sum(), q, create_graph=True)
        terms["eikonal"] = eikonal_term(gradient)
        weights["eikonal"] = config.eikonal_weight
    total = torch.stack([weights[name] * term for name, term in terms.items()]).sum()
    return total, {name: term.detach() for name, term in terms.items()}


def mean_squared_error(predictions: Tensor, expectations: Tensor) -> Tensor:
    return ((predictions - expectations) ** 2).mean()


def eikonal_term(gradient: Tensor) -> Tensor:
    return ((torch.linalg.vector_norm(gradient, dim=-1) - 1) ** 2).mean()


def kr_term(signed: Tensor) -> Tensor:
    return (-signed).mean()


def hinge_term(signed: Tensor, margin: float) -> Tensor:
    return torch.relu(margin - signed).mean()
