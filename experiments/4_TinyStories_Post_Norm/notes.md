# LM Training: Post-Norm instead of Pre-Norm

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

Modal setup is the same as in the [H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup).

To train with post-norm, replace two model files with the post-norm versions in this folder:

- `src/transformer/transformer_block.py` → [`transformer_block_post_norm.py`](transformer_block_post_norm.py), which applies each RMS norm after the residual add (`x = ln1(x + attn(x))`) instead of to the sublayer input (`x = x + attn(ln1(x))`).
- `src/transformer/transformer_lm.py` → [`transformer_lm_post_norm.py`](transformer_lm_post_norm.py), which drops the final RMS norm before the LM head, since the last block's output is already normalized.

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

Apart from the norm placement, these match the best pre-norm setting from the H100 runs.

A sample command to run experiments on Modal GPUs is:
```sh
uv run modal run --detach experiments/4_TinyStories_Post_Norm/experiment_set0.py
```

## Experiment sets

I ran 3 sets of experiments, each building on the previous one. Only `max_lr` is tuned.

| Set | Goal | Iters | Tokens processed |
|---|---|---|---|
| 0 | Coarse `max_lr` sweep | 200 | 39,321,600 |
| 1 | Finer `max_lr` search at double length | 400 | 78,643,200 |
| 2 | Full-length runs at the assignment's H100 token budget | 1700 | 334,233,600 |

Tokens processed = iters × `batch_size` × `context_length`.

Val loss in the tables below is the validation loss at the last iteration, measured on one random batch of 768 sequences.

### Set 0: initial exploration

A coarse sweep over `max_lr` from 0.001 to 0.03, centered on 0.008, the best pre-norm value.

| Run | `max_lr` | Val loss |
|---|---|---|
| set0-0 | 0.008 | 1.92800 |
| set0-1 | 0.01 | 1.97050 |
| set0-2 | 0.03 | 5.80212 |
| set0-3 | 0.003 | 1.98669 |
| set0-4 | 0.001 | 2.44287 |
| set0-5 | 0.009 | 1.99517 |
| set0-6 | 0.006 | **1.92323** |
| set0-7 | 0.007 | 1.93724 |

- The loss is lowest at 0.006 and 0.008 (1.923 and 1.928). 0.007, 0.009 and 0.01 are 0.01–0.07 worse, with no clear trend.
- 0.001 and 0.03 are much worse, and 0.003 is 0.06 worse than the best.

### Set 1: finer learning rate

Set 1 doubles the length to 400 iterations and fills in `max_lr` from 0.002 to 0.01 in steps of 0.001.

| Run | `max_lr` | Val loss |
|---|---|---|
| set1-0 | 0.004 | 1.65018 |
| set1-1 | 0.006 | 1.65032 |
| set1-2 | 0.008 | 1.66383 |
| set1-3 | 0.01 | 5.80807 |
| set1-4 | 0.009 | **1.64105** |
| set1-5 | 0.002 | 1.75992 |
| set1-6 | 0.003 | 1.69537 |
| set1-7 | 0.005 | 1.65408 |
| set1-8 | 0.007 | 1.65757 |

- The loss is flat (1.64–1.66) for `max_lr` from 0.004 to 0.009.
- 0.01 breaks down, ending at 5.81.
- Below 0.004 the loss rises: 1.70 at 0.003 and 1.76 at 0.002.

### Set 2: full-length training

Set 2 trains for 1700 iterations, the same length as [set 2 of the pre-norm H100 runs](../1_TinyStories_Base_H100/notes.md#set-2-full-length-training), with `max_lr` from 0.002 to 0.008. In those pre-norm runs, `max_lr` = 0.008 was optimal (set2-1, val loss 1.33039).

| Run | `max_lr` | Val loss |
|---|---|---|
| set2-0 | 0.004 | **1.38490** |
| set2-1 | 0.006 | 1.39035 |
| set2-2 | 0.008 | 1.41747 |
| set2-3 | 0.002 | 1.39865 |

- **Best run:** set2-0 (`max_lr` = 0.004) has the lowest val loss, 0.055 above the pre-norm optimum.
- **At the pre-norm optimum** `max_lr` = 0.008, post-norm is the worst of the four runs, 0.087 above pre-norm.
- **Optimal `max_lr`** is lower than for pre-norm. The loss is flat from 0.002 to 0.006 (1.385–1.399), and rises at 0.008.
- **Post-norm vs. pre-norm:** all four post-norm runs (1.385–1.417) are worse than all five pre-norm set 2 runs (1.330–1.371), so the gap is larger than the run-to-run noise seen there.
