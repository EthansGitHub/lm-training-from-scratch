import os
import typing

import numpy as np
import torch


def load_data(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    x_len = len(x)
    if x_len <= context_length:
        raise Exception("The input x array must be longer than the input context_length.")

    input_starts = np.random.randint(0, x_len - context_length, batch_size)
    # input_starts = np.arange(batch_size) * context_length
    input_starts_to_target_ends_ids = input_starts[:, None] + np.arange(context_length + 1)
    batch = torch.as_tensor(x[input_starts_to_target_ends_ids], device=device, dtype=torch.long)

    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    data = {
        "model": model.state_dict(),
        "iteration": iteration,
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    data = torch.load(src, map_location=torch.device("cpu"))
    model.load_state_dict(data["model"])
    if optimizer is not None:
        if "optimizer" not in data:
            raise Exception("This checkpoint has no optimizer state.")
        optimizer.load_state_dict(data["optimizer"])
    return data["iteration"]
