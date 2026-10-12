from collections.abc import Iterable
import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None: # type: ignore
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if not state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                t = state["t"]
                m = state["m"]
                v = state["v"]

                param.mul_(1 - lr * weight_decay)

                m.lerp_(param.grad, 1 - beta1)
                v.lerp_(param.grad ** 2, 1 - beta2)

                adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                param.subtract_(adjusted_lr * m / (v.sqrt() + eps))

                state["t"] += 1
