import argparse
import json

from beartype import beartype

from vollo_model_zoo.vm import (
    CONFIGS,
    DEFAULT_CONFIG,
    EXPERIMENTAL_CONFIG_MODEL_COMBOS,
    EXPERIMENTAL_MODELS,
    Ok,
    default_config,
    get_models,
    get_results,
    to_dict,
)


@beartype
def print_table_row(row: list[str]) -> None:
    print("| " + " | ".join(row) + " |")


@beartype
def make_type(choices: list[str]):
    @beartype
    def find_choice(choice: str) -> str:
        for key, item in enumerate([choice.lower() for choice in choices]):
            if choice.lower() == item:
                return choices[key]
        else:
            return choice

    return find_choice


@beartype
def main() -> int:
    """
    Entry point for 'zoo' command. Supports running a specific model.
    """
    available_models = get_models()

    parser = argparse.ArgumentParser(
        description="Run (compute) latency simulations for Vollo models"
    )

    parser.add_argument(
        "model",
        type=make_type(available_models),
        choices=available_models,
        help="Model to run",
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIGS.keys()),
        default=None,
        help=f"Hardware configuration (FPGA) to simulate (default: {DEFAULT_CONFIG},"
        " or the model's own config if it has an experimental one)",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow running experimental models and configs",
    )

    args = parser.parse_args()
    config = args.config if args.config is not None else default_config(args.model)

    if not check_experimental(args.model, config, args.experimental):
        return 1

    return run_model(args.model, config, args.json)


@beartype
def check_experimental(model: str, config: str, experimental: bool) -> bool:
    """
    Check if the requested model or config is experimental, and return whether the run is allowed
    """
    needs_flag = False
    supported = True

    if model in EXPERIMENTAL_MODELS:
        print(f"Note: the '{model}' model is experimental (requires --experimental)")
        needs_flag = True

    if (supported_model := EXPERIMENTAL_CONFIG_MODEL_COMBOS.get(config)) is not None:
        print(
            f"Note: the '{config}' config is experimental (requires --experimental)"
            f" and is only available for the '{supported_model}' model"
        )
        needs_flag = True
        supported = model == supported_model

    if needs_flag and not (experimental and supported):
        print(f"If you are interested in {model} please contact Myrtle to find out")
        print("about upcoming improvements")
        return False

    return True


@beartype
def run_model(model: str, config: str, use_json: bool) -> int:
    # This is a generator
    results = get_results(model, config)

    if use_json:
        print(json.dumps({model: [to_dict(r) for r in results]}))
        return 0

    print(f"VM results for model '{model}' with {config} config:\n")

    headers = [
        "Parameters (M)",
        "Cycles",
        "Latency (us)",
        "Latency contiguous (us)",
        " ".join("" for _ in range(50)) + "Metadata",
    ]

    # This generates a markdown table
    print_table_row(headers)
    print_table_row([f"{'':-<{len(h) - 1}}:" for h in headers])

    for r in results:
        if not isinstance(r, Ok):
            continue

        # Metadata is optional
        meta = {} if r.meta is None else r.meta

        row = [
            f"{r.param_count / 1e6:4.1f}",
            f"{r.cycle_count}",
            f"{r.latency_spaced.microseconds:4.1f}",
            f"{r.latency_contiguous.microseconds:4.1f}",
            ",".join(f"{k}={v}" for k, v in meta.items() if not k.startswith("_")),
        ]

        # Pad each cell to the width of the header
        row = [x.rjust(len(h)) for x, h in zip(row, headers)]

        print_table_row(row)

    print(
        "\nNote: These latencies are from a (near cycle-accurate) software model but without IO (non-negligible for some models)"
    )

    print(
        "\nTip: this is human readable output; use -j/--json for machine-readable output."
    )

    return 0


if __name__ == "__main__":
    exit(main())
