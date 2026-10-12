# LM Training: Varying Batch Size

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one A100-80GB or A100-40GB per run.

Modal setup is the same as in the [H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup). To run:

```sh
uv run modal run --detach experiments/2_TinyStories_Batch_Size/experiment.py
```

## Common hyperparameters

Shared by every run:

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
| Validation | Frequency | every 25 iterations |

## Results

Every run trains on the same 39,321,600 tokens (≈ 39.3M), so `num_train_iters` = 39,321,600 / (`batch_size` × 256). The exception is batch size 1: the full budget would take 153,600 iterations, so to save GPU cost it stops at 19,200 iterations (4,915,200 tokens, 1/8 of the budget).

`max_lr` is retuned for each batch size, and each row is the run with the optimal `max_lr` for that batch size.

| Batch size | Run | `max_lr` | Iters | Tokens processed | GPU | Peak memory (GiB) | Runtime | Val loss |
|---|---|---|---|---|---|---|---|---|
| 768 | set0-0 | 0.008 | 200 | 39,321,600 | A100-80GB | 74.6 | 3 min 46 sec | 1.90238 |
| 512 | set0-3 | 0.008 | 300 | 39,321,600 | A100-80GB | 49.8 | 3 min 50 sec | 1.74403 |
| 256 | set0-19 | 0.007 | 600 | 39,321,600 | A100-40GB | 25.0 | 4 min 34 sec | 1.64743 |
| 128 | set0-32 | 0.005 | 1200 | 39,321,600 | A100-40GB | 12.7 | 5 min 2 sec | **1.62919** |
| 64 | set0-36 | 0.003 | 2400 | 39,321,600 | A100-40GB | 6.5 | 5 min 42 sec | 1.63252 |
| 32 | set0-40 | 0.002 | 4800 | 39,321,600 | A100-40GB | 3.4 | 6 min 22 sec | 1.64402 |
| 8 | set0-48 | 0.001 | 19200 | 39,321,600 | A100-40GB | 1.1 | 29 min 36 sec | 1.75240 |
| 1 | set0-58 | 0.0006 | 19200 | 4,915,200 | A100-40GB | 0.42 | 119 min 6 sec | 2.49811 |

- **Val loss** is the validation loss at the last iteration, averaged over `num_validate_points` random batches of `batch_size` sequences. `num_validate_points` is set so that every run evaluates 768 sequences, except batch size 512, which uses 2 × 512 = 1024.
- **Runtime** is the total wall-clock training time, including the validation passes every 25 iterations.

Training loss, validation loss and gradient norm for all these runs are plotted in [`batch_size.html`](batch_size.html).

## Observations

- **Val loss** is flat from batch size 32 to 256 (1.629–1.647), with 128 the lowest. Gaps that small were within run-to-run noise in the H100 runs.
- **Large batches** do worse at this token budget: 1.744 at 512 and 1.902 at 768. They take only 300 and 200 optimizer steps, and the fixed 150-iteration warmup uses up half and three quarters of those steps.
- **Small batches** also do worse: 1.752 at batch size 8, with the same tokens and 16× the steps of batch size 128.
- **Optimal `max_lr`** grows with batch size, roughly as the square root from 8 to 512 (0.001 → 0.008).
