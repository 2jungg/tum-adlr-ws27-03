"""Fitting a field to the collision labels of one scene.

The loss is the one of the Co-SDF MLP baseline (Sec. IV-A1): MSE between d(q) and the labels
in {-1, +1}, plus an Eikonal term. No Lipschitz constraint, no distance labels.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from metacosdf.config_utils import Config


@dataclass
class TrainingConfig(Config):
    epochs: int = 100
    batch_size: int = 200
    learning_rate: float = 1e-3
    # weight of the Eikonal term mean((|grad_q d(q)| - 1)^2); 0 turns it off
    eikonal_weight: float = 0.1
    # seeds the network initialization and the order of the batches
    seed: int = 0
    # "cpu" or "cuda"
    device: str = "cpu"
    # print the loss every log_every epochs; 0 prints nothing
    log_every: int = 10


def field_loss(field: nn.Module, q: Tensor, labels: Tensor, eikonal_weight: float) -> Tensor:
    """MSE between d(q) and the labels, plus eikonal_weight * mean((|grad_q d(q)| - 1)^2)."""
    if eikonal_weight == 0:
        return ((field(q) - labels) ** 2).mean()
    q = q.detach().requires_grad_(True)
    distance = field(q)
    # create_graph: the Eikonal term itself is differentiated in the backward pass
    (gradient,) = torch.autograd.grad(distance.sum(), q, create_graph=True)
    eikonal = ((torch.linalg.vector_norm(gradient, dim=-1) - 1) ** 2).mean()
    return ((distance - labels) ** 2).mean() + eikonal_weight * eikonal


def fit_field(
    field: nn.Module, q: Tensor, labels: Tensor, config: TrainingConfig
) -> list[float]:
    """
    Train `field` with Adam on the configurations q (N, n) and their labels (N,) in {-1, +1}.
    Returns the mean loss of every epoch.
    """
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
