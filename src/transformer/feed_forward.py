import torch
import torch.nn as nn

from src.transformer.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None) -> None:
        super().__init__()

        if d_ff is None:
            # d_ff = d_model * 8 / 3, then round to the closest multiple of 64
            d_ff = round(d_model / 24) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
