import argparse
import importlib.metadata
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from beartype import beartype
from tqdm import tqdm

from vollo_model_zoo.vm import CONFIGS, get_models, get_results, to_dict


@beartype
def main() -> int:
    """
    Entry point for 'benchmark' command.
    """
    try:
        version = importlib.metadata.version("vollo-compiler")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    default_output = f"./benchmarks/vollo_{version}.json"

    parser = argparse.ArgumentParser(description="Run all models and configurations")

    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"JSON output file (default: {default_output})",
    )

    args = parser.parse_args()

    if Path(args.output).exists():
        print(f"Error: Output file '{args.output}' already exists")
        return 1

    return run_benchmark(args.output, version)


@beartype
def run_benchmark(output_path: str, version: str) -> int:
    models = get_models()
    configs = list(CONFIGS.keys())
    results = defaultdict(dict)

    for model, conf in tqdm(
        product(models, configs),
        desc=f"Benchmarking models (Vollo {version})",
        total=len(models) * len(configs),
    ):
        try:
            results[model][conf] = list(map(to_dict, get_results(model, conf)))
        except Exception as e:
            results[model][conf] = [{"Error": str(e)}]

    data = {version: results}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
