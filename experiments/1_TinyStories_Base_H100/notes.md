# LM Training on H100

## Setup

- **Tokenizer vocab and training data:** TinyStoriesV2-GPT4-train
- **Validation data:** TinyStoriesV2-GPT4-valid
- **Hardware:** Modal, one H100 SXM5 (80 GB) per run

## Modal setup

One-time setup, run from the repo root:

1. Create a Modal secret named `wandb` holding the W&B API key on Modal's website, or using CLI:

   ```sh
   uv run modal secret create wandb WANDB_API_KEY=<your-api-key>
   ```

2. Create the data and checkpoint volumes:

   ```sh
   uv run modal volume create lm-pretraining-data
   uv run modal volume create lm-pretraining-checkpoints
   ```

3. Upload the tokenized training and validation data (created with `src/tokenizer/tokenizer_throughput.py --to_output True`):

   ```sh
   uv run modal volume put lm-pretraining-data data/TinyStoriesV2-GPT4-train-ids.npy /TinyStoriesV2-GPT4-train-ids.npy
   uv run modal volume put lm-pretraining-data data/TinyStoriesV2-GPT4-valid-ids.npy /TinyStoriesV2-GPT4-valid-ids.npy
   ```

Checkpoints are saved at the root of `lm-pretraining-checkpoints` as `checkpoint-<run_name>.pt`. To download one:

```sh
uv run modal volume get lm-pretraining-checkpoints /checkpoint-set2-4.pt experiments/1_TinyStories_Base_H100/checkpoint-set2-4.pt
```

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
| | `min_lr` | 0.1 × `max_lr` |
| | Cosine annealing cycle | ends at the last training iteration (`cosine_cycle_iters` = `num_train_iters`) |
| | Gradient clipping (max L2 norm) | 1 |
| Training | `batch_size` | 768 |
| Validation | `num_validate_points` | 1 |

The model and optimizer match the Mac M1 runs, except for the batch size.

A sample script and command to run experiments on Modal GPUs is:
```sh
uv run modal run --detach experiments/1_TinyStories_Base_H100/experiment_set0.py
```

## Experiment sets

I ran 3 sets of experiments, each building on the previous one.

| Set | Goal | Iters | Tokens processed |
|---|---|---|---|
| 0 | Coarse sweep over `max_lr`, `warmup_iters`, `weight_decay`, `beta2` | 200 | 39,321,600 |
| 1 | Finer `max_lr` and `warmup_iters` search | 300 | 58,982,400 |
| 2 | Full-length runs at the assignment's H100 token budget | 1700 | 334,233,600 |

Tokens processed = iters × `batch_size` × `context_length`.

Val loss in the tables below is the validation loss at the last iteration, measured on one random batch of 768 sequences.

Training loss, validation loss and gradient norm for all these runs are plotted in [`hyper_tuning.html`](hyper_tuning.html).

### Set 0: initial exploration

A coarse sweep around set0-0, which uses the best Mac M1 setting (`max_lr` = 0.003, `warmup_iters` = 100, `weight_decay` = 0.1, `beta2` = 0.95). Each of the other runs changes one hyperparameter from set0-0: runs 1–4 vary `max_lr`, runs 5–6 `warmup_iters`, runs 7–8 `weight_decay`, and runs 9–11 `beta2`.

| Run | `max_lr` | `warmup_iters` | `weight_decay` | `beta2` | Val loss |
|---|---|---|---|---|---|
| set0-0 | 0.003 | 100 | 0.1 | 0.95 | 2.00050 |
| set0-1 | 0.001 | 100 | 0.1 | 0.95 | 2.44996 |
| set0-2 | 0.01 | 100 | 0.1 | 0.95 | **1.90754** |
| set0-3 | 0.03 | 100 | 0.1 | 0.95 | 2.59536 |
| set0-4 | 0.1 | 100 | 0.1 | 0.95 | 3.45199 |
| set0-5 | 0.003 | 50 | 0.1 | 0.95 | 2.03848 |
| set0-6 | 0.003 | 25 | 0.1 | 0.95 | 2.22138 |
| set0-7 | 0.003 | 100 | 0.01 | 0.95 | 2.00999 |
| set0-8 | 0.003 | 100 | 0.03 | 0.95 | 2.00786 |
| set0-9 | 0.003 | 100 | 0.1 | 0.9 | 1.99338 |
| set0-10 | 0.003 | 100 | 0.1 | 0.99 | 2.06581 |
| set0-11 | 0.003 | 100 | 0.1 | 0.999 | 2.04567 |

- **`max_lr`** matters most. The loss is lowest at 0.01 and rises steeply on both sides, so the best value is between 0.003 and 0.03.
- **`warmup_iters`** below 100 hurts: 2.22 at 25, 2.04 at 50, 2.00 at 100.
- **`weight_decay`** barely matters.
- **`beta2`** = 0.95 is good enough. 0.9 is only marginally better, and 0.99 and 0.999 are slightly worse.

### Set 1: finer learning rate and warmup

Set 1 fills in `max_lr` from 0.003 to 0.03 with `warmup_iters` = 100, then tries longer warmups (150 and 200) at the best learning rates. `weight_decay` = 0.1 and `beta2` = 0.95 for all runs.

| Run | `max_lr` | `warmup_iters` | Val loss |
|---|---|---|---|
| set1-0 | 0.02 | 100 | 2.07706 |
| set1-1 | 0.007 | 100 | 1.70711 |
| set1-2 | 0.01 | 150 | 1.69343 |
| set1-3 | 0.01 | 200 | 1.69791 |
| set1-4 | 0.01 | 100 | 1.71437 |
| set1-5 | 0.03 | 100 | 2.23732 |
| set1-6 | 0.003 | 100 | 1.75289 |
| set1-7 | 0.005 | 100 | 1.73309 |
| set1-8 | 0.015 | 100 | 1.90486 |
| set1-9 | 0.013 | 100 | 1.85739 |
| set1-10 | 0.006 | 100 | 1.73506 |
| set1-11 | 0.008 | 100 | 1.69911 |
| set1-12 | 0.009 | 100 | 1.70423 |
| set1-13 | 0.007 | 150 | 1.70333 |
| set1-14 | 0.008 | 150 | **1.67966** |
| set1-15 | 0.009 | 150 | 1.69282 |

- With 100 warmup iterations, the loss is flat (1.70–1.71) for `max_lr` from 0.007 to 0.01 and climbs quickly above 0.01.
- 150 warmup iterations beat 100 at every learning rate tried (0.007–0.01). 200 was no better than 150 at 0.01.

### Set 2: full-length training

Set 2 trains for 1700 iterations, slightly above the assignment's H100 budget of 327,680,000 tokens, with `max_lr` from 0.007 to 0.009. `warmup_iters` = 150, `weight_decay` = 0.1, and `beta2` = 0.95 for all runs.

| Run | `max_lr` | Val loss |
|---|---|---|
| set2-0 | 0.007 | 1.34843 |
| set2-1 | 0.008 | **1.33039** |
| set2-2 | 0.009 | 1.33648 |
| set2-3 | 0.0075 | 1.37074 |
| set2-4 | 0.0085 | 1.36047 |

set2-1 (`max_lr` = 0.008) has the lowest val loss. But set2-3 and set2-4 both did worse than the runs on either side of them, so gaps of a few hundredths here are likely run-to-run noise.

## Resource usage

- **Peak GPU memory:** 74.6 GiB
- **Wall time:** ~17 min per set 2 run

## Sample output

Text generated by set2-4 (I forgot to save set2-1's checkpoint), with these decoding settings:

- **Prompt:** "Once upon a time, there was a"
- **Max tokens:** 256, prompt included (generation also stops at the end-of-text token)
- **Temperature:** 0.4
- **Top-p:** 0.9

To reproduce, run `decode.py` from the repo root. It runs on the CPU. Sampling isn't seeded, so a rerun may give different text.

```sh
uv run python experiments/1_TinyStories_Base_H100/decode.py
```

Output:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her dog, Max. One day, Lily found a big, red ball in her yard. She was so happy!
>
> Lily wanted to show Max the ball. She ran inside and said, "Max, look at my big, red ball!" Max wagged his tail and barked. Lily threw the ball and Max ran to catch it. They played together all day long.
>
> As the sun went down, Lily's mom called her inside. "Lily, it's time to go inside and get ready for bed!" Lily said. Her mom smiled and said, "Okay, Lily. Let's go inside and have a nice nap." They went into their cozy beds and Lily closed her eyes. Max was a very happy dog.
