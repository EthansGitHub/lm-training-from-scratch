# LM Training: Removing the SwiGLU Gate

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

Modal setup is the same as in the [H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup).

To train without the gate, replace `src/transformer/feed_forward.py` with [`feed_forward_remove_gate.py`](feed_forward_remove_gate.py). The feed-forward layer then computes `w2(silu(w1(x)))` (SiLU) instead of `w2(silu(w1(x)) * w3(x))` (SwiGLU). To keep the parameter count about the same, `d_ff` goes from 1344 (≈ 8/3 × `d_model`) to 2048 (4 × `d_model`), giving 2,097,152 parameters per feed-forward layer instead of 2,064,384.

## Common hyperparameters

Shared by every run in all three sets:

| Group | Setting | Value |
|---|---|---|
| Model | `vocab_size` | 10000 |
| | `context_length` | 256 |
| | `d_model` | 512 |
| | `num_layers` | 4 |
| | `num_heads` | 16 |
| Optimizer (AdamW) | `beta1` | 0.9 |
| | `beta2` | 0.95 |
| | `weight_decay` | 0.1 |
| | `warmup_iters` | 150 |
| | `min_lr` | 0.1 × `max_lr` |
| | Cosine annealing cycle | ends at the last training iteration (`cosine_cycle_iters` = `num_train_iters`) |
| | Gradient clipping (max L2 norm) | 1 |
| Training | `batch_size` | 768 |
| Validation | `num_validate_points` | 1 |

Apart from the feed-forward layer, these match the best SwiGLU setting from the H100 runs.

A sample command to run experiments on Modal GPUs is:
```sh
uv run modal run --detach experiments/6_TinyStories_Remove_Gate/experiment_set0.py
```

## Experiment sets

I ran 3 sets of experiments, each building on the previous one. Only `max_lr` is tuned.

| Set | Goal | Iters | Tokens processed |
|---|---|---|---|
| 0 | Quick `max_lr` check around the SwiGLU optimum | 200 | 39,321,600 |
| 1 | Wider `max_lr` sweep at double length | 400 | 78,643,200 |
| 2 | Full-length runs at the assignment's H100 token budget | 1700 | 334,233,600 |

Tokens processed = iters × `batch_size` × `context_length`.

Val loss in the tables below is the validation loss at the last iteration, measured on one random batch of 768 sequences.

### Set 0: initial exploration

Three runs around 0.008, the best `max_lr` with SwiGLU.

| Run | `max_lr` | Val loss |
|---|---|---|
| set0-0 | 0.008 | 2.07382 |
| set0-1 | 0.01 | **2.05646** |
| set0-2 | 0.006 | 2.10270 |

- The loss falls as `max_lr` rises, with 0.01 the best, so set 1 extends the sweep in both directions.

### Set 1: wider learning rate sweep

Set 1 doubles the length to 400 iterations and sweeps `max_lr` from 0.001 to 0.03.

| Run | `max_lr` | Val loss |
|---|---|---|
| set1-0 | 0.008 | 1.64600 |
| set1-1 | 0.01 | **1.64538** |
| set1-2 | 0.03 | 1.89591 |
| set1-3 | 0.003 | 1.79703 |
| set1-4 | 0.001 | 2.12758 |
| set1-5 | 0.006 | 1.67517 |
| set1-6 | 0.02 | 1.82121 |
| set1-7 | 0.009 | 1.65467 |
| set1-8 | 0.007 | 1.65934 |

- The loss is flat (1.645–1.659) for `max_lr` from 0.007 to 0.01, and 0.006 is 0.03 worse.
- The loss rises on both sides: 1.80 at 0.003 and 2.13 at 0.001, and 1.82 at 0.02 and 1.90 at 0.03.

### Set 2: full-length training

Set 2 trains for 1700 iterations, the same length as [set 2 of the H100 runs with SwiGLU](../1_TinyStories_Base_H100/notes.md#set-2-full-length-training), with `max_lr` from 0.007 to 0.01. In those runs with SwiGLU, `max_lr` = 0.008 was optimal (set2-1, val loss 1.33039).

| Run | `max_lr` | Val loss |
|---|---|---|
| set2-0 | 0.007 | 1.34975 |
| set2-1 | 0.008 | 1.35852 |
| set2-2 | 0.009 | **1.33597** |
| set2-3 | 0.01 | 1.34404 |

- **Best run:** set2-2 (`max_lr` = 0.009) has the lowest val loss, 0.006 above the optimum with SwiGLU.
- **Optimal `max_lr`** is about the same as with SwiGLU. The loss spans 1.336–1.359 across 0.007–0.01, and 0.008 is worse than both of its neighbors, so these gaps are likely run-to-run noise.
- **SiLU vs. SwiGLU:** the four SiLU runs (1.336–1.359) fall inside the range of the five SwiGLU set 2 runs (1.330–1.371). At this scale, removing the gate makes no difference larger than the run-to-run noise.

## Resource usage

- **Peak GPU memory:** 76.8 GiB, vs. 74.6 GiB with SwiGLU
