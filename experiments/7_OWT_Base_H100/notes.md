# LM Training on H100

## Setup

- **Tokenizer vocab and training data:** owt_train
- **Validation data:** owt_valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

## Modal setup

The W&B secret and the data and checkpoint volumes are the same as in the [TinyStories H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup).

Upload the tokenized training and validation data (created with `src/tokenizer/tokenizer_throughput.py --to_output True`):

```sh
uv run modal volume put lm-pretraining-data data/owt_train-ids.npy /owt_train-ids.npy
uv run modal volume put lm-pretraining-data data/owt_valid-ids.npy /owt_valid-ids.npy
```

## Common hyperparameters

Shared by every run in sets 0–3:

| Group | Setting | Value |
|---|---|---|
| Model | `vocab_size` | 32000 |
| | `context_length` | 256 |
| | `d_model` | 512 |
| | `num_layers` | 4 |
| | `num_heads` | 16 |
| Optimizer (AdamW) | `beta1` | 0.9 |
| | `min_lr` | 0.1 × `max_lr` |
| | Cosine annealing cycle | ends at the last training iteration (`cosine_cycle_iters` = `num_train_iters`) |
| | Gradient clipping (max L2 norm) | 1 |
| Validation | `num_validate_points` | 768 / `batch_size` |

The model and optimizer match the TinyStories H100 runs, except for the vocab size and the batch size.

A sample command to run experiments on Modal GPUs is:
```sh
uv run modal run --detach experiments/7_OWT_Base_H100/experiment_set0.py
```

## Experiment sets

The four sets below each build on the previous one.

| Set | Goal | `batch_size` | Iters | Tokens processed |
|---|---|---|---|---|
| 0 | Batch size at a fixed token count | 64–256 | 600–2400 | 39,321,600 |
| 1 | Sweep over `warmup_iters`, `weight_decay`, `beta2` | 128 | 1200 | 39,321,600 |
| 2 | `max_lr` search | 128 | 1200 | 39,321,600 |
| 3 | Full-length runs, same token budget as the [TinyStories H100 runs](../1_TinyStories_Base_H100/notes.md) | 128 | 10200 | 334,233,600 |

Tokens processed = iters × `batch_size` × `context_length`.

Val loss in the tables below is the validation loss at the last iteration, averaged over `num_validate_points` random batches of `batch_size` sequences (768 sequences in total).

Training loss, validation loss and gradient norm for all these runs are plotted in [`hyper_tuning.html`](hyper_tuning.html).

### Set 0: batch size

Set 0 trains every run on the same 39,321,600 tokens, so `num_train_iters` = 39,321,600 / (`batch_size` × 256). `max_lr` = 0.004, `warmup_iters` = 150, `weight_decay` = 0.1, and `beta2` = 0.95 for all runs.

| Run | `batch_size` | Iters | `num_validate_points` | Val loss |
|---|---|---|---|---|
| set0-0 | 256 | 600 | 3 | 4.63820 |
| set0-1 | 128 | 1200 | 6 | 4.53029 |
| set0-2 | 64 | 2400 | 12 | 4.52266 |

- Batch size 256 is clearly worse (4.638).
- 64 and 128 are within 0.008 of each other. However, 64's runtime is about 20% longer than 128's, so later sets use 128.

### Set 1: warmup, weight decay and beta2

A sweep around set0-1 (`max_lr` = 0.004, `warmup_iters` = 150, `weight_decay` = 0.1, `beta2` = 0.95). Each run changes one hyperparameter from set0-1: runs 0–1 vary `warmup_iters`, runs 2–4 `beta2`, and runs 5–9 `weight_decay`. set0-1 is repeated in the table for comparison.

| Run | `warmup_iters` | `weight_decay` | `beta2` | Val loss |
|---|---|---|---|---|
| set0-1 | 150 | 0.1 | 0.95 | 4.53029 |
| set1-0 | 100 | 0.1 | 0.95 | 4.54602 |
| set1-1 | 200 | 0.1 | 0.95 | 4.51280 |
| set1-2 | 150 | 0.1 | 0.9 | 4.56579 |
| set1-3 | 150 | 0.1 | 0.99 | 4.60460 |
| set1-4 | 150 | 0.1 | 0.999 | 4.56750 |
| set1-5 | 150 | 0.01 | 0.95 | 4.56645 |
| set1-6 | 150 | 0.03 | 0.95 | 4.52553 |
| set1-7 | 150 | 0.2 | 0.95 | 4.51898 |
| set1-8 | 150 | 0.05 | 0.95 | 4.57689 |
| set1-9 | 150 | 0.08 | 0.95 | 4.54963 |

- **`warmup_iters`**: the loss drops slightly with longer warmup: 4.546 at 100, 4.530 at 150 and 4.513 at 200 (the lowest in the set).
- **`beta2`** = 0.95 is the best. 0.9 and 0.999 are about 0.04 worse (4.566 and 4.568), and 0.99 is the worst run in the set (4.605).
- **`weight_decay`** shows no trend: 4.566 at 0.01, 4.526 at 0.03, 4.577 at 0.05, 4.550 at 0.08, 4.530 at 0.1 and 4.519 at 0.2.
- All runs fall between 4.51 and 4.61. The `weight_decay` results go up and down by as much as 0.05 with no clear direction, so gaps of a few hundredths are likely run-to-run noise. Sets 2 and 3 keep `warmup_iters` = 150, `weight_decay` = 0.1 and `beta2` = 0.95.

### Set 2: learning rate

Set 2 tries `max_lr` from 0.002 to 0.005. `warmup_iters` = 150, `weight_decay` = 0.1, and `beta2` = 0.95 for all runs. set0-1 (`max_lr` = 0.004) is repeated in the table for comparison.

| Run | `max_lr` | Val loss |
|---|---|---|
| set0-1 | 0.004 | 4.53029 |
| set2-0 | 0.002 | 4.60160 |
| set2-1 | 0.003 | 4.52617 |
| set2-2 | 0.005 | 4.55116 |
| set2-3 | 0.0025 | 4.56259 |
| set2-4 | 0.0035 | 4.55617 |

- The loss is lowest at 0.003 (4.526), with 0.004 close behind (4.530). Only 0.002 is clearly worse (4.602).
- 0.0025, 0.0035 and 0.005 all land between 4.55 and 4.56. 0.0035 did worse than both 0.003 and 0.004, so the loss is flat within noise from 0.0025 to 0.005.

### Set 3: full-length training

Set 3 trains for 10200 iterations (334,233,600 tokens). This is the same token budget as the full-length runs in the [TinyStories H100 notes](../1_TinyStories_Base_H100/notes.md#set-2-full-length-training) (1700 iterations × 768 sequences there, 10200 × 128 here), slightly above the assignment's H100 budget of 327,680,000 tokens. It compares the two best learning rates from set 2, `max_lr` = 0.003 and 0.004. `warmup_iters` = 150, `weight_decay` = 0.1, and `beta2` = 0.95 for both runs.

| Run | `max_lr` | Val loss |
|---|---|---|
| set3-0 | 0.003 | **3.91314** |
| set3-1 | 0.004 | 3.92762 |

set3-0 (`max_lr` = 0.003) has the lower val loss. The gap (0.014) is within the spread that looked like noise in sets 1 and 2, so the two learning rates are effectively tied.

Because the validation loss is large, text generation is unstable.

## Resource usage

- **Peak GPU memory:** 21 GiB
- **Wall time:** ~1400 sec (~23 min) per set 3 run
