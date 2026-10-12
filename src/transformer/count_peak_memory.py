import json

from src.utils import format, count_parameters


def run(vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int, num_heads: int) -> None:
    parameters = count_parameters(vocab_size, num_layers, d_model, d_ff)["total"] * 4
    gradients = parameters
    optimizer_state = 2 * parameters
    activation_slope = 2 * context_length * vocab_size * 4
    activation_bias = (context_length ** 2 + 2 * d_model / num_heads) * num_layers * 4

    peak_memory = {
        "parameters_per_batch": parameters,
        "gradients_per_batch": gradients,
        "optimizer_state_per_batch": optimizer_state,
        "activation_slope": activation_slope,
        "activation_bias": activation_bias,
    }
    peak_memory["total_slope"] = parameters + gradients + optimizer_state + activation_slope
    peak_memory["total_bias"] = activation_bias

    format_peak_memory = format(peak_memory)
    format_peak_memory["activation"] = format_peak_memory["activation_slope"] + " * batch_size + " + format_peak_memory["activation_bias"]
    format_peak_memory["total"] = format_peak_memory["total_slope"] + " * batch_size + " + format_peak_memory["total_bias"]

    for k in ("activation_slope", "activation_bias", "total_slope", "total_bias"):
        del format_peak_memory[k]

    print(json.dumps({"peak_memory": format_peak_memory}, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)

    args = parser.parse_args()

    run(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
    )
