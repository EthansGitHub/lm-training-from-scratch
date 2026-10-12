# Tokenizer Experiments

Compression ratio and throughput of the tokenizer on different datasets.

System: MacBook Air M1, 16 GB memory.

TinyStoriesV2-GPT4-train has 541 M tokens, while owt_train has 2.73 B tokens.

## Summary

### Compression ratio

| Vocab Source | Sample Source | Sample Number | Compression Ratio |
|---|---|---|---|
| TinyStoriesV2-GPT4-train | TinyStoriesV2-GPT4-train | 10 | 4.16 bytes/token |
| owt_train | owt_train | 10 | 4.70 bytes/token |
| TinyStoriesV2-GPT4-train | owt_train | 10 | 3.20 bytes/token |
| owt_train | TinyStoriesV2-GPT4-train | 10 | 4.06 bytes/token |

### Throughput

| Data | LRU Cache Capacity | Throughput |
|---|---|---|
| owt_train | 10,000 | 1.85 MB/sec |

## Runs

Run the following commands in the root folder.

### Compression ratio: TinyStories vocab, TinyStories sample

```bash
uv run python src/tokenizer/tokenizer_comp_ratio.py \
  --vocab_source TinyStoriesV2-GPT4-train \
  --sample_source TinyStoriesV2-GPT4-train \
  --sample_number 10
```

**Result:** 4.16 bytes per token

### Compression ratio: OWT vocab, OWT sample

```bash
uv run python src/tokenizer/tokenizer_comp_ratio.py \
  --vocab_source owt_train \
  --sample_source owt_train \
  --sample_number 10
```

**Result:** 4.70 bytes per token

### Compression ratio: TinyStories vocab, OWT sample

```bash
uv run python src/tokenizer/tokenizer_comp_ratio.py \
  --vocab_source TinyStoriesV2-GPT4-train \
  --sample_source owt_train \
  --sample_number 10
```

**Result:** 3.20 bytes per token

### Compression ratio: OWT vocab, TinyStories sample

```bash
uv run python src/tokenizer/tokenizer_comp_ratio.py \
  --vocab_source owt_train \
  --sample_source TinyStoriesV2-GPT4-train \
  --sample_number 10
```

**Result:** 4.06 bytes per token

### Throughput: owt_train

```bash
uv run python src/tokenizer/tokenizer_throughput.py \
  --vocab_data_file_prefix owt_train \
  --input_data_file_prefix owt_train \
  --lru_cache_capacity 10000 \
  --to_output True
```

**Result:** 1.85 MB per second

### Tokenize owt_valid

```bash
uv run python src/tokenizer/tokenizer_throughput.py \
  --vocab_data_file_prefix owt_train \
  --input_data_file_prefix owt_valid \
  --lru_cache_capacity 10000 \
  --to_output True
```
