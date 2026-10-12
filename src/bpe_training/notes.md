# BPE Training Benchmarks

Runtime and peak memory of training BPE on different datasets. System: MacBook Air M1, 16 GB memory.

## Summary

| Dataset | Workers | Vocab Size | Time | Peak Memory |
|---|---|---|---|---|
| TinyStoriesV2-GPT4-valid | 1 | 10,000 | 3 sec | 36 MB |
| TinyStoriesV2-GPT4-train | 8 | 10,000 | 59 sec | 120 MB |
| owt_valid | 8 | 32,000 | 21 sec | 1.15 GB |
| owt_train | 8 | 32,000 | 525 sec | 12.78 GB |

## Runs

Run the following commands in the root folder.

### TinyStoriesV2-GPT4-valid

```bash
/usr/bin/time -l uv run python src/bpe_training/run_bpe_trainer.py \
  --n_workers 1 \
  --vocab_size 10000 \
  --data_file_prefix TinyStoriesV2-GPT4-valid \
  --to_debug False \
  --to_output False
```

**Result:** 3 sec, peak memory 36 MB

### TinyStoriesV2-GPT4-train

```bash
/usr/bin/time -l uv run python src/bpe_training/run_bpe_trainer.py \
  --n_workers 8 \
  --vocab_size 10000 \
  --data_file_prefix TinyStoriesV2-GPT4-train \
  --to_debug True \
  --to_output True
```

**Result:** 59 sec, peak memory 120 MB

Use `cProfile` to pinpoint the bottleneck of time efficiency:
```bash
uv run python -m cProfile -s tottime src/bpe_training/run_bpe_trainer.py \
  --n_workers 1 \
  --vocab_size 10000 \
  --data_file_prefix TinyStoriesV2-GPT4-train \
  --to_debug False \
  --to_output False | head -n 30
```
**Result:** 452 sec  
**Bottleneck:** inner logic of `BPETrainer._find_word_counts`, `str.encode` and `_regex.Match`.

### owt_valid

```bash
/usr/bin/time -l uv run python src/bpe_training/run_bpe_trainer.py \
  --n_workers 8 \
  --vocab_size 32000 \
  --data_file_prefix owt_valid \
  --to_debug False \
  --to_output False
```

**Result:** 21 sec, peak memory 1.15 GB

### owt_train

```bash
/usr/bin/time -l uv run python src/bpe_training/run_bpe_trainer.py \
  --n_workers 8 \
  --vocab_size 32000 \
  --data_file_prefix owt_train \
  --to_debug True \
  --to_output True
```

**Result:** 525 sec, peak memory 12.78 GB
