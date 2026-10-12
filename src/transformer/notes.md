# Memory and FLOPs

## Task 1

Run the following commands in the root folder to get the number of parameters, total memory of these parameters (assume float32), the total number of FLOPs resulting from matrix multiplication in a forward pass.

### GPT-2 small

```bash
uv run python src/transformer/count_memory_and_flops.py \
  --vocab_size 50257 \
  --context_length 1024 \
  --num_layers 12 \
  --d_model 768 \
  --d_ff 2048
```

### GPT-2 medium

```bash
uv run python src/transformer/count_memory_and_flops.py \
  --vocab_size 50257 \
  --context_length 1024 \
  --num_layers 24 \
  --d_model 1024 \
  --d_ff 2752
```

### GPT-2 large

```bash
uv run python src/transformer/count_memory_and_flops.py \
  --vocab_size 50257 \
  --context_length 1024 \
  --num_layers 36 \
  --d_model 1280 \
  --d_ff 3392
```

### GPT-2 XL

```bash
uv run python src/transformer/count_memory_and_flops.py \
  --vocab_size 50257 \
  --context_length 1024 \
  --num_layers 48 \
  --d_model 1600 \
  --d_ff 4288
```


## Task 2

Run the following commands in the root folder to get the peak memory.

### GPT-2 XL

```bash
uv run python src/transformer/count_peak_memory.py \
  --vocab_size 50257 \
  --context_length 1024 \
  --num_layers 48 \
  --d_model 1600 \
  --d_ff 4288 \
  --num_heads 25
```
