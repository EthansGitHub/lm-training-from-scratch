# LM Training: Removing RMS Norm

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

Modal setup is the same as in the [H100 notes](../1_TinyStories_Base_H100/notes.md#modal-setup).

To train without RMS norm, replace two model files with the no-norm versions in this folder:

- `src/transformer/transformer_block.py` → [`transformer_block_remove_norm.py`](transformer_block_remove_norm.py), which drops the RMS norms before attention and before the feed-forward layer.
- `src/transformer/transformer_lm.py` → [`transformer_lm_remove_norm.py`](transformer_lm_remove_norm.py), which drops the final RMS norm before the LM head.

Then run:

```sh
uv run modal run --detach experiments/3_TinyStories_Remove_Norm/experiment.py
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
| Training | `batch_size` | 768 |
| | `num_train_iters` | 1700 (334,233,600 tokens) |
| Validation | Frequency | every 50 iterations, on one batch of 768 sequences |

## Results

- **With-the-RMS-norm `max_lr`:** `max_lr` = 0.008, the best value for the model with RMS norm, blows up at iteration 111. Training loss jumps to about 13,500 and validation loss to about 1.5e14.
- **Higher `max_lr`:** every `max_lr` ≥ 0.002 I tried blew up before iteration 1700.
- **Best run:** `max_lr` = 0.001 is the best among the runs that finished, with validation loss 1.53372 at iteration 1700.

Training loss, validation loss and gradient norm for these two runs are plotted in [`remove_norm.html`](remove_norm.html).
