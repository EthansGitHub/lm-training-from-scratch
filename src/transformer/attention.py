import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from src.transformer.linear import Linear
from src.transformer.rope import RotaryPositionalEmbedding
from src.utils import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    score = einsum(Q / math.sqrt(Q.shape[-1]), K, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2")
    if mask is not None:
        score.masked_fill_(~mask, -torch.inf)

    return einsum(softmax(score, -1), V, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")


def create_mask(max_seq_len: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)).bool()


def create_token_positions(max_seq_len: int, device=None) -> torch.Tensor:
    return torch.arange(max_seq_len, device=device)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        mask: torch.Tensor | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = rope
        self.mask = mask

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        rearrange_pattern = "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k"
        Q = rearrange(self.q_proj(x), rearrange_pattern, num_heads=self.num_heads, d_k=self.d_k)
        K = rearrange(self.k_proj(x), rearrange_pattern, num_heads=self.num_heads, d_k=self.d_k)
        V = rearrange(self.v_proj(x), rearrange_pattern, num_heads=self.num_heads, d_k=self.d_k)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        if self.mask is None:
            mask = create_mask(seq_len, device=x.device)
        else:
            mask = self.mask[..., :seq_len, :seq_len]

        attention = rearrange(
            scaled_dot_product_attention(Q, K, V, mask=mask),
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )

        return self.output_proj(attention)
