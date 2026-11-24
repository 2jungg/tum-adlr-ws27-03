import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from metacosdf.config_utils import Config
from metacosdf.losses import LossConfig, field_loss


@dataclass
class TrainingConfig(Config):
    # Co-SDF, 2-DoF: Adam, learning rate 5e-4, batch 200, 200 epochs
    epochs: int = 200
    batch_size: int = 200
    learning_rate: float = 5e-4
    # network initialization and batch order
    seed: int = 0
    device: str = "cpu"
    # 0: no printing
    log_every: int = 10


def fit_field(
    field: nn.Module,
    q: Tensor,
    labels: Tensor,
    loss_config: LossConfig,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    """q (N, n), labels (N,) -> per epoch, the means of the total loss and of every term"""
    device = torch.device(config.device)
    field.to(device)
    q, labels = q.to(device), labels.to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=config.learning_rate)
    batch_order = torch.Generator().manual_seed(config.seed)
    history = []
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(len(q), generator=batch_order).to(device)
        # summed on the device: an .item() per batch would make the CPU wait for the GPU
        sums: defaultdict[str, Tensor] = defaultdict(lambda: torch.zeros((), device=device))
        for batch in permutation.split(config.batch_size):
            total, terms = field_loss(field, q[batch], labels[batch], loss_config, epoch)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for name, value in {"total": total.detach(), **terms}.items():
                sums[name] += value * len(batch)
        # the only wait for the GPU in the epoch
        means = (torch.stack(list(sums.values())) / len(q)).tolist()
        history.append(dict(zip(sums, means, strict=True)))
        if not math.isfinite(history[-1]["total"]):
            raise FloatingPointError(f"epoch {epoch}: non-finite loss {history[-1]}")
        if config.log_every and epoch % config.log_every == 0:
            terms_text = ", ".join(f"{name} {value:.4f}" for name, value in history[-1].items())
            print(f"epoch {epoch:4d}/{config.epochs}: {terms_text}")
    return history
