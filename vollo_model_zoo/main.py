import argparse
import importlib
import json
import os
from collections.abc import Generator
from dataclasses import asdict
from typing import Optional
from warnings import warn

from beartype import beartype
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo.vm import CONFIGS, Result


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
    available_models = get_available_models()

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
        default="V80",
        help="Hardware configuration (FPGA) to simulate",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    results = get_model_results(args.model, args.config)

    if args.json:
        print(json.dumps({args.model: [asdict(r) for r in results]}))
        return 0

    print(f"VM results for model '{args.model}' with {args.config} config:\n")

    headers = [
        "Parameters (M)",
        "Cycles",
        "Latency/us",
        "Latency/us (contiguous)",
        " ".join("" for _ in range(50)) + "Metadata",
    ]

    # This generates a markdown table
    print_table_row(headers)
    print_table_row([f"{'':-<{len(h) - 1}}:" for h in headers])

    try:
        for r in results:
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
    except (AllocationError, SaveError):
        warn("Some configurations don's fit on this config")

    print(
        "\nTip: this is human readable output; use -j/--json for machine-readable output."
    )

    return 0


@beartype
def get_available_models() -> list[str]:
    """
    Parse the models directory to find available models. Each model should be a
    .py file not starting with __ (to exclude __init__.py, etc.). The model
    name is derived from the filename.
    """
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    @beartype
    def is_valid_model_file(filename: str) -> bool:
        return filename.endswith(".py") and not filename.startswith("__")

    return sorted(f[:-3] for f in os.listdir(models_dir) if is_valid_model_file(f))


@beartype
def get_model_results(
    model_name: str, config: Optional[str]
) -> Generator[Result, None, None]:
    """
    Import the specified model module and call its main() function, this is
    expected to be a generator that yields Result objects.
    """
    model_module_path = f"vollo_model_zoo.models.{model_name}"
    model_module = importlib.import_module(model_module_path)

    if not hasattr(model_module, "main"):
        raise ImportError(
            f"Model module '{model_module_path}' does not have a callable main() function."
        )

    if config is None:
        return model_module.main()
    else:
        return model_module.main(config=config)


@beartype
def print_table_row(row: list[str]) -> None:
    print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    exit(main())
