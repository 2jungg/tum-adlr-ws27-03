from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def torchify_as(reference_tensor: Tensor) -> Callable[[Any], Tensor]:
    def torchify(target_data: Any) -> Tensor:
        return torch.as_tensor(
            data=target_data, dtype=reference_tensor.dtype, device=reference_tensor.device
        )

    return torchify
