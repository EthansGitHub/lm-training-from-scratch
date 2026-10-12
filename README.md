# LM Pretraining from Scratch

I built a small GPT-style language model from scratch in PyTorch and pretrained it on TinyStories and OpenWebText. The project covers the whole pipeline: a byte-level BPE tokenizer, the Transformer, the AdamW optimizer, and the training loop. I then ran a series of experiments on batch size, normalization, positional encoding, and the feed-forward gate.

**Highlights**

- **TinyStories:** validation loss **1.33** after 334M training tokens, in about 17 minutes on one H100. On a MacBook Air M1 with a 41M-token budget it reaches 1.61.
- **OpenWebText:** validation loss **3.91** with the same 334M-token budget, in about 23 minutes on one H100.
- **Ablations on TinyStories** (baseline 1.33): post-norm gets 1.385, removing RoPE 1.402, and removing RMSNorm 1.534 (training diverges at any learning rate above 0.001). Replacing SwiGLU with a parameter-matched SiLU gets 1.336, which is within run-to-run noise.

## Repository layout

```text
.
├── src/                  # all model, tokenizer and training code
│   ├── bpe_training/     # BPE tokenizer training
│   ├── tokenizer/        # encoding / decoding with a trained vocab
│   ├── transformer/      # model components and the full LM
│   ├── lm_training/      # optimizer, training loop, checkpointing, decoding
│   └── utils.py
├── experiments/          # one folder per experiment
├── tests/                # unit tests; adapters.py connects them to src/
├── assignment_instructions.pdf
├── pyproject.toml
└── uv.lock
```

> **Each part has its own `notes.md`.** `src/bpe_training/`, `src/tokenizer/`, `src/transformer/`, and every folder in `experiments/` include a `notes.md` with the exact commands, hyperparameters, results tables, and observations. This README gives the overview; the notes have the details.

## Setup

### Environment

I manage the environment with [uv](https://docs.astral.sh/uv/). Install it, then run this from the repo root:

```sh
uv sync
```

This creates `.venv/` with Python 3.12 or 3.13 and installs:

- **Runtime:** PyTorch, einops, NumPy, regex, Weights & Biases
- **Dev group** (included by default): pytest, jaxtyping, Modal, tiktoken, ty

Run any script with `uv run`, and run the unit tests with:

```sh
uv run pytest
```

For training runs, you may also need:

- **Weights & Biases:** the experiment scripts log to W&B. Run `uv run wandb login` once.
- **Modal** (cloud GPUs): run `uv run modal setup` once, then do the one-time secret and volume setup in [experiments/1_TinyStories_Base_H100/notes.md](experiments/1_TinyStories_Base_H100/notes.md#modal-setup).

Local experiments use the `mps` backend on Apple silicon. GPU experiments run on Modal (H100 or A100).

### Data

Download TinyStories and a subsample of OpenWebText into `./data/`.

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Pipeline

Run every command from the repo root. The scripts read from and write to `./data/`. The examples here use TinyStories; for OpenWebText, use the `owt_train` / `owt_valid` prefixes and a 32,000-token vocab.

1. **Train the BPE tokenizer.** This writes `data/<prefix>-vocab.pkl` and `data/<prefix>-merges.pkl`.

   ```sh
   uv run python src/bpe_training/run_bpe_trainer.py \
     --n_workers 8 \
     --vocab_size 10000 \
     --data_file_prefix TinyStoriesV2-GPT4-train \
     --to_debug False \
     --to_output True
   ```

2. **Tokenize the training and validation text.** This writes `data/<prefix>-ids.npy`, an array of `uint16` token IDs. Run it once with `TinyStoriesV2-GPT4-train` as the input and once with `TinyStoriesV2-GPT4-valid`:

   ```sh
   uv run python src/tokenizer/tokenizer_throughput.py \
     --vocab_data_file_prefix TinyStoriesV2-GPT4-train \
     --input_data_file_prefix TinyStoriesV2-GPT4-train \
     --lru_cache_capacity 10000 \
     --to_output True
   ```

3. **Train the model.** Each experiment folder has a launch script.

   ```sh
   # locally, on Apple silicon
   uv run python experiments/0_TinyStories_Base_Mac_M1/experiment_set0.py

   # on a Modal H100
   uv run modal run --detach experiments/1_TinyStories_Base_H100/experiment_set0.py
   ```

4. **Generate text** from a saved checkpoint:

   ```sh
   uv run python experiments/1_TinyStories_Base_H100/decode.py
   ```

## Source code: `src/`

| Folder | What's inside | `notes.md` |
|---|---|---|
| [`bpe_training/`](src/bpe_training) | Byte-level BPE trainer. It splits the corpus into chunks at special-token boundaries for regex pre-tokenization, then picks merges using a max-heap and incremental pair-count updates. `run_bpe_trainer.py` is the CLI. | Runtime and peak memory on each dataset. |
| [`tokenizer/`](src/tokenizer) | Tokenizer built from a trained vocab and merges. It handles special tokens, streams input through `encode_iterable`, and uses an LRU cache for repeated words. Includes scripts that measure compression ratio and throughput. | Compression ratio for each vocab and dataset pair, and encoding throughput |
| [`transformer/`](src/transformer) | `Linear`, `Embedding`, `RMSNorm`, RoPE, causal multi-head self-attention, the SwiGLU feed-forward layer, a pre-norm Transformer block, and the full `TransformerLM`. Includes scripts that count parameters, FLOPs, and peak memory. | Parameter, FLOP, and memory accounting for GPT-2 small through XL |
| [`lm_training/`](src/lm_training) | AdamW, cross-entropy, gradient clipping, a cosine learning-rate schedule with warmup, batch sampling from token arrays, and checkpointing. `LMTrainer` runs the training and validation loop. `experiment()` is the entry point every experiment script calls. `Decoder` does temperature and top-p sampling. | — |
| [`utils.py`](src/utils.py) | Softmax, parameter counting, and a timing decorator | — |

## Experiments: `experiments/`

Unless an experiment changes it, every model has 4 layers, `d_model` = 512, 16 heads, context length 256, a SwiGLU feed-forward layer with `d_ff` = 1344, and untied input and output embeddings. That comes to about 23M parameters with the 10k TinyStories vocab and about 45M with the 32k OpenWebText vocab. Each experiment tunes the learning rate (and sometimes other hyperparameters) over progressively longer sets of runs, then compares at full length.

| Folder | What it tests | Hardware | Best val loss |
|---|---|---|---|
| [`0_TinyStories_Base_Mac_M1`](experiments/0_TinyStories_Base_Mac_M1/notes.md) | Baseline and hyperparameter sweep on a laptop (41M tokens) | MacBook Air M1 | 1.606 |
| [`1_TinyStories_Base_H100`](experiments/1_TinyStories_Base_H100/notes.md) | Same model at batch size 768 with a 334M-token budget | H100 | **1.330** |
| [`2_TinyStories_Batch_Size`](experiments/2_TinyStories_Batch_Size/notes.md) | Batch size from 1 to 768 with a fixed 39M tokens, retuning the learning rate for each | A100 | 1.629 |
| [`3_TinyStories_Remove_Norm`](experiments/3_TinyStories_Remove_Norm/notes.md) | Remove every RMSNorm | H100 | 1.534 |
| [`4_TinyStories_Post_Norm`](experiments/4_TinyStories_Post_Norm/notes.md) | Post-norm instead of pre-norm | H100 | 1.385 |
| [`5_TinyStories_Remove_RoPE`](experiments/5_TinyStories_Remove_RoPE/notes.md) | No positional encoding (the causal mask is the only position signal) | H100 | 1.402 |
| [`6_TinyStories_Remove_Gate`](experiments/6_TinyStories_Remove_Gate/notes.md) | SiLU feed-forward instead of SwiGLU, parameter-matched | H100 | 1.336 |
| [`7_OWT_Base_H100`](experiments/7_OWT_Base_H100/notes.md) | Baseline on OpenWebText with a 32k vocab and the 334M-token budget | H100 | 3.913 |

**Takeaways**

- Pre-norm, RoPE, and RMSNorm each clearly help. Without any normalization, training diverges at any learning rate above 0.001.
- The SwiGLU gate made no difference beyond run-to-run noise at this scale.
- With a fixed token budget, batch sizes from 32 to 256 do about equally well. The best learning rate grows roughly with the square root of the batch size.

**What's in each experiment folder**

- `notes.md`: the setup, shared hyperparameters, every run's validation loss, observations, and resource usage. The notes for folders 0 and 1 also include sample generated stories.
- `*.html`: interactive plots of training loss, validation loss, and gradient norm for every run. GitHub doesn't render these, so download them and open them in a browser.
- `experiment*.py`: a launch script for a sweep.
- Ablation folders (3–6) include modified copies of model or trainer files. To reproduce an ablation, swap them into `src/` as described in that folder's notes.
- ``decode.py`: a script to generate text from a saved checkpoint file.

## Acknowledgments

This project started as Assignment 1 of [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu) (Spring 2026). The original handout is [assignment_instructions.pdf](assignment_instructions.pdf).

I wrote all the Python code in `src/` and `experiments/`. Everything else, including the unit tests, test snapshots, and project configuration, comes from the course starter code with small changes and is covered by the MIT license in [LICENSE](LICENSE). Most of the Markdown write-ups were drafted with AI assistance.
