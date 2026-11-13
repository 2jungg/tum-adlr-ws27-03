"""C-space fields: the contract every field follows, and network fields d(q) = g(phi(q)).

The contract: q (B, n) in radians -> d (B,), positive in free space.
Ground truths, dummies and networks all follow it,
so evaluation and plotting work the same for all of them.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from metacosdf.robot import PlanarArm
from metacosdf.scene_generator.scene import cspace_grid

type Field = Callable[[Tensor], Tensor]


def embed(robot: PlanarArm, q: Tensor) -> Tensor:
    """
    (..., n) -> (..., 2n): phi(q), the input of every network.
    The centers of the links (Co-SDF, Sec. III-C): periodic in q,
    and the network sees where the robot's body is.
    """
    return robot.embedding(q)


def embedding_size(robot: PlanarArm) -> int:
    """The size of phi(q): two coordinates per link."""
    return 2 * robot.dof


class NeuralField(nn.Module):
    """d(q) = network(phi(q)): the network gets the embedding, never raw joint angles."""

    def __init__(self, robot: PlanarArm, network: nn.Module) -> None:
        super().__init__()
        self.robot = robot
        self.network = network

    def forward(self, q: Tensor) -> Tensor:
        """(..., n) -> (...)"""
        return self.network(embed(self.robot, q))


@torch.no_grad()
def evaluate_on_grid(
    field: Field, resolution: int = 256, device: str = "cpu", chunk_size: int = 16384
) -> Tensor:
    """(res, res) values of a field on cspace_grid(res), computed in chunks, on the CPU."""
    q = cspace_grid(resolution).reshape(-1, 2).to(device)
    values = torch.cat([field(q_chunk) for q_chunk in q.split(chunk_size)])
    return values.reshape(resolution, resolution).cpu()
