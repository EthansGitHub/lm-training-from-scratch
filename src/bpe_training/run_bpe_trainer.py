from src.bpe_training.bpe_trainer import BPETrainer


def run(n_workers: int, vocab_size: int, data_file_prefix: str, to_debug: bool, to_output: bool) -> None:
    input_path = f"./data/{data_file_prefix}.txt"

    client = None
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        ProcessPoolExecutor(max_workers=n_workers)

    trainer = BPETrainer(n_workers, client)
    vocab, merges = trainer.train(input_path, vocab_size, ["<|endoftext|>"])

    if to_output:
        import pickle

        vocab_output_path = f"./data/{data_file_prefix}-vocab.pkl"
        merges_output_path = f"./data/{data_file_prefix}-merges.pkl"

        with open(vocab_output_path, "wb") as f:
            pickle.dump(vocab, f)

        with open(merges_output_path, "wb") as f:
            pickle.dump(merges, f)

    if to_debug:
        import json

        debug_vocab_output_path = f"./debug/{data_file_prefix}-vocab.json"
        debug_merges_output_path = f"./debug/{data_file_prefix}-merges.json"

        debug_vocab = {}
        for i, b in vocab.items():
            try:
                debug_vocab[i] = b.decode("utf-8")
            except Exception:
                debug_vocab[i] = str(b)

        with open(debug_vocab_output_path, "w") as f:
            json.dump(debug_vocab, f, indent=4)

        debug_merges = []
        for bp in merges:
            try:
                debug_merges.append((bp[0].decode("utf-8"), bp[1].decode("utf-8")))
            except Exception:
                debug_merges.append((str(bp[0]), str(bp[1])))

        with open(debug_merges_output_path, "w") as f:
            json.dump(debug_merges, f, indent=4)


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--data_file_prefix", type=str, required=True)
    parser.add_argument("--to_debug", type=ast.literal_eval, required=True)
    parser.add_argument("--to_output", type=ast.literal_eval, required=True)

    args = parser.parse_args()

    run(
        n_workers=args.n_workers,
        vocab_size=args.vocab_size,
        data_file_prefix=args.data_file_prefix,
        to_debug=args.to_debug,
        to_output=args.to_output,
    )
