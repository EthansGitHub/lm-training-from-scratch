from src.tokenizer.tokenizer import Tokenizer


def run(vocab_source: str, sample_source: str, sample_number: int) -> None:
    input_path = f"./data/{sample_source}.txt"

    mini_chunk_size = 4096
    pos = 0
    count = 1
    with open(input_path, "rb") as file:
        while count <= sample_number:
            file.seek(pos)
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                break

            found_at = mini_chunk.find(b"<|endoftext|>")
            if found_at != -1:
                pos += found_at + 13
                count += 1
                continue

            # -13 in case a mini chunk stops in the middle of <|endoftext|>
            pos += mini_chunk_size - 13

        if count > sample_number:
            pos -= 13
        else:
            raise Exception(f"{sample_source} does not have {sample_number} docs.")

        file.seek(0)
        text_bytes = file.read(pos)

    text_byte_count = len(text_bytes)

    text = text_bytes.decode("utf-8")

    vocab_output_path = f"./data/{vocab_source}-vocab.pkl"
    merges_output_path = f"./data/{vocab_source}-merges.pkl"

    tok = Tokenizer.from_files(vocab_output_path, merges_output_path)
    ids = tok.encode(text)
    token_count = len(ids)

    print(f"Docs: {sample_source} {sample_number} docs")
    print(f"Vocab: {vocab_source}")
    print(f"Compression ratio (bytes per token): {text_byte_count / token_count}")
    print(f"Round-trip recovery: {text == tok.decode(ids)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_source", type=str, required=True)
    parser.add_argument("--sample_source", type=str, required=True)
    parser.add_argument("--sample_number", type=int, required=True)

    args = parser.parse_args()

    run(
        vocab_source=args.vocab_source,
        sample_source=args.sample_source,
        sample_number=args.sample_number,
    )
