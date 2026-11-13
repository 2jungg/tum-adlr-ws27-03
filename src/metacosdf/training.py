"""Fitting a field to collision labels, with the loss of the Co-SDF MLP baseline."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from metacosdf.config_utils import Config


@dataclass
class TrainingConfig(Config):
    epochs: int = 100
    batch_size: int = 200
    learning_rate: float = 1e-3
    # 0 turns the Eikonal term off
    eikonal_weight: float = 0.1
    # network initialization and batch order
    seed: int = 0
    device: str = "cpu"
    # 0: no printing
    log_every: int = 10


def field_loss(field: nn.Module, q: Tensor, labels: Tensor, eikonal_weight: float) -> Tensor:
    """MSE to the labels + eikonal_weight * mean((|grad_q d(q)| - 1)^2)"""
    if eikonal_weight == 0:
        return ((field(q) - labels) ** 2).mean()
    q = q.detach().requires_grad_(True)
    distance = field(q)
    # create_graph: the Eikonal term is differentiated too
    (gradient,) = torch.autograd.grad(distance.sum(), q, create_graph=True)
    eikonal = ((torch.linalg.vector_norm(gradient, dim=-1) - 1) ** 2).mean()
    return ((distance - labels) ** 2).mean() + eikonal_weight * eikonal


def fit_field(
    field: nn.Module, q: Tensor, labels: Tensor, config: TrainingConfig
) -> list[float]:
    """q (N, n), labels (N,) in {-1, +1} -> mean loss of every epoch"""
    device = torch.device(config.device)
    field.to(device)
    q, labels = q.to(device), labels.to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=config.learning_rate)
    batch_order = torch.Generator().manual_seed(config.seed)
    history = []
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(len(q), generator=batch_order).to(device)
        total_loss = 0.0
        for batch in permutation.split(config.batch_size):
            loss = field_loss(field, q[batch], labels[batch], config.eikonal_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        history.append(total_loss / len(q))
        if config.log_every and epoch % config.log_every == 0:
            print(f"epoch {epoch:4d}/{config.epochs}: loss {history[-1]:.4f}")
    return history
