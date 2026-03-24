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

    mk_default_output = "./benchmarks/README.md"
    parser.add_argument(
        "--markdown_output",
        type=Path,
        default=mk_default_output,
        help=f"Markdown output file for summary table (default: {mk_default_output})",
    )

    args = parser.parse_args()

    return run_benchmark(args, version)


@beartype
def run_benchmark(args: argparse.Namespace, version: str) -> int:
    models = get_models()
    configs = list(CONFIGS.keys())
    results = defaultdict(dict)

    if args.json_output.exists():
        print(f"Error: JSON output file '{args.json_output}' already exists")
        # return 1
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)

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

    with open(args.json_output, "w") as f:
        json.dump(data, f, indent=2)

    # Produce markdown summary table

    with open(args.markdown_output, "w") as f:
        print(f"# Vollo Model Zoo Benchmarks (Vollo {version})\n", file=f)

        print(
            "Compute latency for an approximately 1-million parameter model.\n", file=f
        )

        for config in configs:
            print(f"## Configuration: {config}\n", file=f)

            print("| Model | Latency/us | Latency/us (contiguous) | Metadata |", file=f)
            print("|-------|------------|-------------------------|----------|", file=f)

            for model in models:
                # All versions
                variants = [x["Ok"] for x in results[model][config] if "Ok" in x]
                # The one closest to 1-mil parameters
                chosen = min(variants, key=lambda x: abs(x["param_count"] - 1_000_000))

                l1 = chosen["latency_spaced"]["microseconds"]
                l2 = chosen["latency_contiguous"]["microseconds"]
                meta = ",".join(f"{k}={v}" for k, v in chosen.get("meta", {}).items())

                print(f"| {model} | {l1:.2f} | {l2:.2f} | {meta} |", file=f)

    return 0


if __name__ == "__main__":
    exit(main())
