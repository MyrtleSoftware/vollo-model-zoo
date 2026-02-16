import argparse
import importlib
import os

from dataclasses import asdict
import json

from beartype import beartype

from vollo_model_zoo.vm import Result

from typing import Generator


@beartype
def get_available_models() -> list[str]:
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    @beartype
    def is_valid_model_file(filename: str) -> bool:
        return filename.endswith(".py") and not filename.startswith("__")

    return sorted(f[:-3] for f in os.listdir(models_dir) if is_valid_model_file(f))


@beartype
def print_table_row(row: list[str]) -> None:
    print("  | " + " | ".join(row) + " |")


@beartype
def main() -> int:
    available_models = get_available_models()

    parser = argparse.ArgumentParser(
        description="Run (compute) latency simulations for Vollo models"
    )

    parser.add_argument("model", choices=available_models, help="Model to run")

    parser.add_argument(
        "-j", "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    model_module_path = f"vollo_model_zoo.models.{args.model}"

    model_module = importlib.import_module(model_module_path)

    if hasattr(model_module, "main"):
        results: Generator[Result] = model_module.main()
    else:
        raise ImportError(
            f"Model module '{model_module_path}' does not have a main() function."
        )

    if args.json:
        print(json.dumps({args.model: [asdict(r) for r in results]}))
        return 0

    print(f"VM results for model '{args.model}':")

    headers = [
        "Parameters (M)",
        "Cycles",
        "Latency/us (spaced)",
        "Latency/us (back-to-back)",
    ]

    # This generates a markdown table
    print_table_row(headers)
    print_table_row([f"{'':-<{len(h) - 1}}:" for h in headers])

    for r in results:
        row = [
            f"{r.param_count / 1e6:4.1f}",
            f"{r.cycle_count}",
            f"{r.latency_fast.microseconds:4.1f}",
            f"{r.latency_slow.microseconds:4.1f}",
        ]

        # Pad each cell to the width of the header
        row = [x.rjust(len(h)) for x, h in zip(row, headers)]

        print_table_row(row)

    print(
        "Tip: this is human readable output; use -j/--json for machine-readable output."
    )

    return 0


if __name__ == "__main__":
    exit(main())
