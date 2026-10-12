# LM Training: Removing RoPE

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

Modal setup is the same as in the [H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup).

To train without RoPE, replace `src/lm_training/lm_trainer.py` with [`lm_trainer_remove_rope.py`](lm_trainer_remove_rope.py), which passes `rope=None` to the model instead of building a `RotaryPositionalEmbedding`. Attention then skips rotating the queries and keys, so the only position information left is the causal mask.

## Common hyperparameters

Shared by every run in both sets:

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

Apart from removing RoPE, these match the best setting from the H100 runs.

A sample command to run experiments on Modal GPUs is:
```sh
uv run modal run --detach experiments/5_TinyStories_Remove_RoPE/experiment_set0.py
```

## Experiment sets

I ran 2 sets of experiments, the second building on the first. Only `max_lr` is tuned.

| Set | Goal | Iters | Tokens processed |
|---|---|---|---|
| 0 | Coarse `max_lr` sweep | 200 | 39,321,600 |
| 1 | Full-length runs at the assignment's H100 token budget | 1700 | 334,233,600 |

Tokens processed = iters × `batch_size` × `context_length`.

Val loss in the tables below is the validation loss at the last iteration, measured on one random batch of 768 sequences.

### Set 0: initial exploration

A coarse sweep over `max_lr` from 0.001 to 0.03, centered on 0.008, the best value with RoPE.

| Run | `max_lr` | Val loss |
|---|---|---|
| set0-0 | 0.008 | 2.25313 |
| set0-1 | 0.01 | 2.28799 |
| set0-2 | 0.03 | 3.17786 |
| set0-3 | 0.003 | 2.37544 |
| set0-4 | 0.001 | 2.92549 |
| set0-5 | 0.009 | 2.28300 |
| set0-6 | 0.006 | **2.19002** |
| set0-7 | 0.004 | 2.32020 |
| set0-8 | 0.005 | 2.25956 |
| set0-9 | 0.007 | 2.19594 |

- The loss is lowest at 0.006 and 0.007 (2.190 and 2.196). 0.005 and 0.008–0.01 are 0.06–0.10 worse.
- 0.004 and 0.003 are 0.13 and 0.19 worse, and 0.001 and 0.03 are much worse (2.93 and 3.18).

### Set 1: full-length training

Set 1 trains for 1700 iterations, the same length as [set 2 of the H100 runs with RoPE](../1_TinyStories_Base_H100/notes.md#set-2-full-length-training), with `max_lr` from 0.005 to 0.009. In those runs with RoPE, `max_lr` = 0.008 was optimal (set2-1, val loss 1.33039).

| Run | `max_lr` | Val loss |
|---|---|---|
| set1-0 | 0.005 | 1.41573 |
| set1-1 | 0.006 | 1.43354 |
| set1-2 | 0.007 | 1.41015 |
| set1-3 | 0.008 | 1.40885 |
| set1-4 | 0.009 | **1.40233** |
| set1-5 | 0.01 | 1.43111 |

- **Best run:** set1-4 (`max_lr` = 0.009) has the lowest val loss, 0.072 above the optimum with RoPE.
- **Optimal `max_lr`:** the loss is flat from 0.007 to 0.009 (1.402–1.410), around the RoPE optimum of 0.009.
- **Without vs. with RoPE:** all five runs without RoPE (1.402–1.434) are worse than all five H100 set 2 runs with RoPE (1.330–1.371).
