import argparse
from itertools import groupby
from pathlib import Path

from beartype import beartype
from beartype.typing import Optional

from vollo_model_zoo.parse import parse_records_from_json
from vollo_model_zoo.version import describe_version


@beartype
def main() -> int:
    """
    Entry point for 'report' command.
    """
    parser = argparse.ArgumentParser(description="Generate report from benchmarks")

    parser.add_argument(
        "input",
        type=Path,
        help="JSON input file",
    )

    mk_default_output = "./benchmarks/README.md"
    parser.add_argument(
        "--output",
        type=Path,
        default=mk_default_output,
        help=f"Markdown output file for summary table (default: {mk_default_output})",
    )

    parser.add_argument(
        "--plots",
        type=Path,
        nargs="+",
        help="Optional plot image files to include in the report (relative paths)",
    )

    args = parser.parse_args()

    return _generate_report(args.input, args.output, args.plots)


@beartype
def _generate_report(input: Path, output: Path, plots: Optional[list[Path]]) -> int:
    """
    Generate performance plots from benchmark JSON files.
    """
    records = parse_records_from_json(input)

    assert records, "No valid benchmark results found."

    versions = [x["version"] for x in records]

    assert all(
        v == versions[0] for v in versions
    ), "Multiple versions found in input JSON"

    def key_func(x):
        return (x["config"], x["model"])

    records.sort(key=key_func, reverse=True)

    with open(output, "w") as f:

        def write(x):
            print(x, file=f)

        # Format as markdown
        write(f"# Vollo Model Zoo Benchmarks ({describe_version(versions[0])})")
        write("")
        write("Compute latency for an approximately 1-million parameter model.")
        write("")
        write(
            "Note: These latencies are from a (near cycle-accurate) software "
            "model but without IO (non-negligible for some models)"
        )

        for config, config_group in groupby(records, key=lambda x: x["config"]):
            write("")
            write(f"## Configuration: {config}")
            write("")
            write("| Model | Latency (us) | Latency contiguous (us)  | Metadata |")
            write("| ----- | ------------ | ------------------------ | -------- |")

            for model, group in groupby(config_group, key=lambda x: x["model"]):
                # The one closest to 1-mil parameters
                chosen = min(group, key=lambda x: abs(x["param_count"] - 1_000_000))

                l1 = chosen["latency_spaced"]
                l2 = chosen["latency_contiguous"]
                mt = chosen["meta"]

                print(f"| {model} | {l1:.2f} | {l2:.2f} | {mt} |", file=f)

        if not plots:
            return 0

        write("")
        write("## Performance over time")
        write("")
        write("Click to expand each plot:")

        for plot in plots:
            write("")
            write("<details>")
            write(f"<summary>{plot.stem}</summary>")
            write("")
            write(f"![{plot.stem}](../{plot})")
            write("")
            write("</details>")

    return 0


if __name__ == "__main__":
    exit(main())
