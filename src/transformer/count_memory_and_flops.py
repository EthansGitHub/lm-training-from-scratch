import json

from src.utils import format, count_parameters


def run(vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int) -> None:
    parameters = count_parameters(vocab_size, num_layers, d_model, d_ff)

    fp32_memory = {k: 4 * v for k, v in parameters.items()}

    flops = {
        "attention": 4 * num_layers * context_length * d_model * (2 * d_model + context_length),
        "feed_forward": 6 * num_layers * context_length * d_model * d_ff,
        "lm_head": 2 * context_length * vocab_size * d_model,
    }
    flops["total"] = sum(flops.values())

    parameters = format(parameters)
    fp32_memory = format(fp32_memory)
    flops = format(flops)

    print(json.dumps({"parameters": parameters, "fp32_memory": fp32_memory, "flops": flops}, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)

    args = parser.parse_args()

    run(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )
