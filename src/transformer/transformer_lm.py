import torch
import torch.nn as nn

from src.transformer.embedding import Embedding
from src.transformer.linear import Linear
from src.transformer.rms_norm import RMSNorm
from src.transformer.rope import RotaryPositionalEmbedding
from src.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        mask: torch.Tensor | None = None,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, rope=rope, mask=mask, d_ff=d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_final(self.layers(self.token_embeddings(token_ids))))
