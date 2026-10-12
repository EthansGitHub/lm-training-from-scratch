import os
import time

from src.tokenizer.tokenizer import Tokenizer


def run(
    vocab_data_file_prefix: str,
    input_data_file_prefix: str,
    lru_cache_capacity: int,
    to_output: bool
) -> None:
    input_path = f"./data/{input_data_file_prefix}.txt"
    vocab_output_path = f"./data/{vocab_data_file_prefix}-vocab.pkl"
    merges_output_path = f"./data/{vocab_data_file_prefix}-merges.pkl"

    tok = Tokenizer.from_files(vocab_output_path, merges_output_path, lru_cache_capacity=lru_cache_capacity)
    ids = []
    with open(input_path, "r") as file:
        start = time.perf_counter()
        for id in tok.encode_iterable(file):
            ids.append(id)
        lapse = time.perf_counter() - start

    size_bytes = os.path.getsize(input_path)

    print(f"Vocab file: {vocab_data_file_prefix}")
    print(f"Data file: {input_data_file_prefix}")
    print(f"Throughput (MB per second): {size_bytes / lapse / 1024 ** 2}")

    if to_output:
        import numpy as np

        ids_output_path = f"./data/{input_data_file_prefix}-ids.npy"
        np.save(ids_output_path, np.array(ids, dtype=np.uint16))


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_data_file_prefix", type=str, required=True)
    parser.add_argument("--input_data_file_prefix", type=str, required=True)
    parser.add_argument("--lru_cache_capacity", type=int, required=True)
    parser.add_argument("--to_output", type=ast.literal_eval, required=True)

    args = parser.parse_args()

    run(
        vocab_data_file_prefix=args.vocab_data_file_prefix,
        input_data_file_prefix=args.input_data_file_prefix,
        lru_cache_capacity=args.lru_cache_capacity,
        to_output=args.to_output,
    )
