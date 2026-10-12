import torch
import torch.nn as nn

from src.transformer.attention import CausalMultiHeadSelfAttention
from src.transformer.feed_forward import FeedForward
from src.transformer.rms_norm import RMSNorm
from src.transformer.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        mask: torch.Tensor | None = None,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.ffn = FeedForward(d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, mask=mask, device=device, dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
