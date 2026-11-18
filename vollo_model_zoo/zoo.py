import argparse
import json

from beartype import beartype
from tabulate import tabulate

from vollo_model_zoo import catalog, vm
from vollo_model_zoo.vm import (
    CONFIGS,
    DEFAULT_CONFIG,
    Ok,
    get_results,
    to_dict,
)


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
    available_models = catalog.models()

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
        help=f"Hardware configuration (FPGA) to simulate (default: {DEFAULT_CONFIG})",
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
        help="Allow running experimental models, and sweep experimental sizes",
    )

    args = parser.parse_args()
    config = args.config if args.config is not None else DEFAULT_CONFIG

    if not check_experimental(args.model, args.experimental):
        return 1
    vm.sweep_experimental = args.experimental

    return run_model(args.model, config, args.json)


@beartype
def check_experimental(model: str, experimental: bool) -> bool:
    """
    Check if the requested model is experimental, and return whether the run is allowed
    """
    reason = catalog.experimental(model)
    if reason is None:
        return True

    print(f"Note: the '{model}' model is experimental (requires --experimental):")
    print(f"{reason}.")
    if not experimental:
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

    rows = []
    for r in results:
        if not isinstance(r, Ok):
            continue
        meta = {k: v for k, v in (r.meta or {}).items() if not k.startswith("_")}
        rows.append(
            {
                "params (M)": r.param_count / 1e6,
                "compute cycles": r.cycle_count,
                "compute latency (us)": r.latency_spaced.microseconds,
                "contiguous latency (us)": r.latency_contiguous.microseconds,
                **meta,
            }
        )

    # A markdown table, so it pastes into an issue or PR as it is.
    print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))

    print(
        "\nNote: These latencies are from a (near cycle-accurate) software model but without IO (non-negligible for some models)"
    )

    print(
        "\nTip: this is human readable output; use -j/--json for machine-readable output."
    )

    return 0


if __name__ == "__main__":
    exit(main())
