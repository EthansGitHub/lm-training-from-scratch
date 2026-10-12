import time

import numpy as np
import torch

from src.lm_training.adamw import AdamW
from src.transformer.attention import create_mask
from src.lm_training.data_helpers import load_checkpoint, load_data, save_checkpoint
from src.transformer.rope import RotaryPositionalEmbedding
from src.lm_training.training_helpers import cross_entropy, cosine_annealing, clip_gradient
from src.transformer.transformer_lm import TransformerLM


class LMTrainer:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        max_lr: float,
        min_lr: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        weight_decay: float,
        beta1: float,
        beta2: float,
        gradient_ceiling: float,
        train_data_file_prefix: str,
        validate_data_file_prefix: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        rope = RotaryPositionalEmbedding(10000, d_model // num_heads, context_length, device=device)
        mask = create_mask(context_length, device=device)

        self.model = TransformerLM(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            rope=rope,
            mask=mask,
            device=device,
            dtype=dtype,
        )
        if device == "mps":
            self.model.compile()
        else:
            self.model.compile(backend="aot_eager")

        if device == "cuda":
            torch.set_float32_matmul_precision("high")

        self.optimizer = AdamW(self.model.parameters(), min_lr, weight_decay, (beta1, beta2))
        self.train_iter = 0

        self.train_data_file_prefix = train_data_file_prefix
        self.context_length = context_length
        self.device = device

        self._clip_gradient = lambda x: clip_gradient(x, gradient_ceiling)
        self._cosine_annealing = lambda x: cosine_annealing(x, max_lr, min_lr, warmup_iters, cosine_cycle_iters)

        self.train_data = np.load(f"./data/{self.train_data_file_prefix}-ids.npy", mmap_mode="r")

        self.valid_data = None
        if validate_data_file_prefix is not None:
            self.valid_data = np.load(f"./data/{validate_data_file_prefix}-ids.npy", mmap_mode="r")

    @torch.no_grad()
    def validate_loss(self, batch_size: int, num_validate_points: int) -> float:
        if self.valid_data is None:
            raise Exception("Validation data is None.")

        valid_loss = 0
        for _ in range(num_validate_points):
            x_valid, y_valid = load_data(self.valid_data, batch_size, self.context_length, device=self.device)
            valid_loss += cross_entropy(self.model(x_valid), y_valid).item()
        return valid_loss / num_validate_points

    def train_one_iteration(self, batch_size: int) -> tuple[float, ...]:
        start = time.perf_counter()

        self.train_iter += 1

        x_train, y_train = load_data(self.train_data, batch_size, self.context_length, device=self.device)

        self.optimizer.zero_grad()
        train_loss = cross_entropy(self.model(x_train), y_train)
        train_loss.backward()
        grad_norm_pre_clip = self._clip_gradient(self.model.parameters())

        lr = self._cosine_annealing(self.train_iter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.step()

        train_loss = train_loss.item()

        lapse = time.perf_counter() - start

        return train_loss, grad_norm_pre_clip, lapse

    def save_checkpoint(self, path: str = "", include_optimizer: bool = True):
        save_checkpoint(
            self.model,
            self.train_iter,
            path,
            optimizer=self.optimizer if include_optimizer else None,
        )

    def load_checkpoint(self, path: str = "", include_optimizer: bool = True):
        self.train_iter = load_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer if include_optimizer else None,
        )
