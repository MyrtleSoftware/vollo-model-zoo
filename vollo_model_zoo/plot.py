import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from beartype import beartype
from packaging.version import parse as parse_version

from vollo_model_zoo.parse import parse_records_from_json


@beartype
def main() -> int:
    """
    Entry point for 'plot' command.
    """
    parser = argparse.ArgumentParser(description="Generate plots from benchmarks")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON input file(s) (default: benchmarks_*.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    _generate_plots(args.inputs, args.output_dir)

    return 0


@beartype
def _generate_plots(input_files: list[str], output_dir: str):
    """
    Generate performance plots from benchmark JSON files.
    """

    records = []
    for input_file in input_files:
        records.extend(parse_records_from_json(input_file))

    if not records:
        print("No valid benchmark results found.")
        return

    df = pd.DataFrame(records)
    df["version_parsed"] = df["version"].apply(parse_version)
    df = df.sort_values("version_parsed")

    os.makedirs(output_dir, exist_ok=True)

    for (model, config), group in df.groupby(["model", "config"]):
        _plot_config(model, config, group, output_dir)

    print(f"Generated {len(df.groupby(['model', 'config']))} plots in '{output_dir}'.")


@beartype
def _plot_config(model: str, config: str, group: pd.DataFrame, output_dir: str):
    """
    Generate a single plot for a given model and config.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for meta, sub_group in group.groupby("meta"):
        ax1.plot(
            sub_group["version"],
            sub_group["latency_spaced"],
            marker="o",
            label=meta,
        )
        ax2.plot(
            sub_group["version"],
            sub_group["latency_contiguous"],
            marker="o",
            label=meta,
        )

    ax1.set_title("Spaced Latency")
    ax1.set_ylabel("Latency (us)")
    ax2.set_title("Contiguous Latency")
    ax2.set_ylabel("Latency (us)")
    ax2.set_xlabel("Version (Vollo SDK + model zoo)")

    # Composite versions make for long tick labels; the axes are shared, so
    # rotating the bottom one covers both.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f"Performance vs Version\nModel: {model} | Config: {config}")

    if not all(group["meta"] == ""):
        ax1.legend(title="Parameters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    def safe_name(s):
        return s.replace("-", "_").replace(" ", "_")

    plt.savefig(
        os.path.join(output_dir, f"{safe_name(model)}_{safe_name(config)}.svg"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    exit(main())
