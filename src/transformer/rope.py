import torch
import torch.nn as nn
from einops import einsum


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()

        pos_ids = torch.arange(max_seq_len, device=device)
        pos1_angles = theta ** (- torch.arange(0, d_k, 2, device=device) / d_k)
        angles = einsum(pos_ids, pos1_angles, "pos_id, angle_id -> pos_id angle_id")

        self.sin_angles: torch.Tensor
        self.cos_angles: torch.Tensor
        self.register_buffer("sin_angles", torch.sin(angles), persistent=False)
        self.register_buffer("cos_angles", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        d_k = x.shape[-1]
        x_rotated = torch.empty_like(x)

        x_1 = x[..., 0:d_k:2]
        x_2 = x[..., 1:d_k:2]

        if token_positions is None:
            cos_angles = self.cos_angles[:x.shape[-2]].to(x.dtype)
            sin_angles = self.sin_angles[:x.shape[-2]].to(x.dtype)
        else:
            cos_angles = self.cos_angles[token_positions].to(x.dtype)
            sin_angles = self.sin_angles[token_positions].to(x.dtype)

        x_rotated[..., 0:d_k:2] = x_1 * cos_angles - x_2 * sin_angles
        x_rotated[..., 1:d_k:2] = x_1 * sin_angles + x_2 * cos_angles

        return x_rotated
