import math
from collections.abc import Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    entropy_denom = torch.log(logits.exp().sum(dim=-1))
    entropy_num = logits.gather(-1, targets[..., None]).squeeze(-1)
    return (entropy_denom - entropy_num).mean()


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]

    norm = math.sqrt(sum(torch.linalg.norm(g) ** 2 for g in grads))
    if norm >= max_l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (norm + 1e-6))

    return norm

def cosine_annealing(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it <= warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate + \
            0.5 * (1 + \
                   math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * \
                    (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
