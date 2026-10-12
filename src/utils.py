import time
import functools

import torch


def clock(old_func):
    @functools.wraps(old_func)
    def new_func(*args, **kwargs):
        start = time.perf_counter()
        outputs = old_func(*args, **kwargs)
        lapse = time.perf_counter() - start
        print(f"{old_func.__name__} costs {lapse} sec.")
        return outputs
    return new_func


def format(str_int_dict: dict) -> dict:
    return {k: f"{int(v):,}" for k, v in str_int_dict.items()}


def count_parameters(vocab_size: int, num_layers: int, d_model: int, d_ff: int) -> dict:
    parameters = {
        "embedding": vocab_size * d_model,
        "rms_norm": (2 * num_layers + 1) * d_model,
        "attention": 4 * num_layers * d_model ** 2,
        "feed_forward": 3 * num_layers * d_model * d_ff,
        "lm_head": vocab_size * d_model,
    }
    parameters["total"] = sum(parameters.values())
    return parameters


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = (x - x.max(dim, keepdim=True).values).exp()
    return exp_x / exp_x.sum(dim, keepdim=True)