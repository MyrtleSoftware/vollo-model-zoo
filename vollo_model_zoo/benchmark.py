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

    parser = argparse.ArgumentParser(description="Run all models and configurations")

    json_default_output = f"./benchmarks/vollo_{version}.json"
    parser.add_argument(
        "--json_output",
        type=Path,
        default=json_default_output,
        help=f"JSON output file (default: {json_default_output})",
    )

    args = parser.parse_args()

    return run_benchmark(args.json_output, version)


@beartype
def run_benchmark(json_output: Path, version: str) -> int:
    models = get_models()
    configs = list(CONFIGS.keys())
    results = defaultdict(dict)

    if json_output.exists():
        print(f"Error: JSON output file '{json_output}' already exists")
        # return 1
    else:
        json_output.parent.mkdir(parents=True, exist_ok=True)

    for model, config in tqdm(
        product(models, configs),
        desc=f"Benchmarking models (Vollo {version})",
        total=len(models) * len(configs),
    ):
        try:
            results[model][config] = list(map(to_dict, get_results(model, config)))
        except Exception as e:
            results[model][config] = [{"Error": str(e)}]

    data = {version: results}

    with open(json_output, "w") as f:
        json.dump(data, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
