"""
The hardware path of the zoo, driven by `scripts/fpga-benchmark.sh` as
`python -m vollo_model_zoo.fpga <command>`:

- `compile` compiles every size to a `.vollo` program for a hw config read
  from a card, using the public compiler API.
- `collect` turns the compile outputs and the `vollo-example` runs into a
  results JSON in the shape of `benchmarks/*.json`, under the zoo config's
  name suffixed with `FPGA_SUFFIX`.
- `config` prints the zoo config a card's hw config is.

Files in the programs directory, per size `<model>-<size>`:
    .vollo + .num-models + .meta.json   compiled
    .error.json                         didn't compile, as its results entry
and `sizes.txt` lists `<model> <size>` lines. Files in the runs directory, per
model index `<model>-<size>.<index>`: `.contiguous.json`, `.spaced.json` (the
`vollo-example --json` output), `.contiguous.run-error` / `.spaced.run-error`
(a run of that kind failed) or `.skipped` (not run after an earlier run timed
out; a message each).
An index with one run failed is still `Ok`: that run's latency is null and its
side of `hardware` is `{"RunError": msg}`. Only both failing is a `RunError`.
Sizes excluded by --skip are left out entirely.

Latencies here are measured on the card, so they include host<->accelerator IO.
"""

import argparse
import copy
import json
import math
import sys
import traceback
from pathlib import Path

import vollo_compiler as vc
from beartype import beartype
from beartype.typing import Optional, Sequence

from vollo_model_zoo import catalog, vm
from vollo_model_zoo.parse import FPGA_SUFFIX
from vollo_model_zoo.version import benchmark_version


@beartype
def _same_hardware(hw_config: vc.Config, name: str) -> bool:
    """
    Whether `hw_config` is the zoo config `name`, ignoring name and clock.
    """
    zoo_config = copy.copy(vm.CONFIGS[name])
    zoo_config.name = hw_config.name
    zoo_config.clock_mhz = hw_config.clock_mhz
    return hw_config == zoo_config


@beartype
def detect_config(hw_config: vc.Config) -> str:
    """
    The name of the one zoo config a card's hw config is. Never guesses: no
    match, or several, is a ValueError.

    Configs are first matched on hardware alone, as a card's name and clock may
    differ from the zoo's. Some zoo configs are the same hardware
    (`V80` and `V80LL` differ only in clock, `IA-420f` and `NT400D11` only in
    name), so those are told apart by what the card reports: its name, then its
    clock. A tie-break that matches none of them is ignored rather than ruling
    everything out.
    """
    matches = [name for name in vm.CONFIGS if _same_hardware(hw_config, name)]

    for same in (
        lambda config: config.name == hw_config.name,
        lambda config: config.clock_mhz == hw_config.clock_mhz,
    ):
        if len(matches) > 1:
            matches = [name for name in matches if same(vm.CONFIGS[name])] or matches

    if len(matches) != 1:
        raise ValueError(
            f"the card's hw config ({hw_config.name}, {hw_config.num_cores} cores, "
            f"block size {hw_config.block_size}, {hw_config.clock_mhz} MHz) "
            f"matches {matches or 'no zoo config'}; "
            f"pass --config, one of {list(vm.CONFIGS)}"
        )
    return matches[0]


@beartype
def _skipped(skips: Sequence[str], model: str, size: str) -> bool:
    return any(skip in (model, f"{model}-{size}") for skip in skips)


@beartype
def _compile_size(
    model: str, size: str, config: str, hw_config: vc.Config, out: Path
) -> None:
    name = f"{model}-{size}"
    # Outside the `try`: building a size compiles nothing, so whatever it
    # raises is a zoo bug, not a compile result, and propagates as on the VM
    # path. Built for the card's hw config rather than the zoo's `config`, as
    # an `instance` may shape the model by it (e.g. `config.num_cores`).
    zoo = catalog.build(model, size, hw_config)
    try:
        program = vm.compile_instance(zoo, hw_config)
        info = [
            dict(param_count=ok.param_count, cycle_count=ok.cycle_count, meta=ok.meta)
            for ok in vm.program_results(zoo, program, config)
        ]
        (out / f"{name}.num-models").write_text(f"{program.num_models()}\n")
        (out / f"{name}.meta.json").write_text(json.dumps(info))
        # Last: scripts/fpga-benchmark.sh runs every size with a `.vollo`, so
        # it is only there once everything else about the size is.
        program.save(str(out / f"{name}.vollo"))
    except (vm.ZooArgumentError, MemoryError):
        # A size the zoo can't compile, or the machine running out of memory:
        # neither is the SDK's doing, so neither passes for a compile result.
        _unsave(out, name)
        raise
    except (vc.AllocationError, vc.SaveError, ValueError) as e:
        # The failures `vollo_info` returns as values, recorded as the VM
        # benchmarks record them. A ValueError from saving the program is
        # recorded the same way, where the VM path would let it propagate.
        print(f"{name}: {type(e).__name__}: {e}")
        _unsave(out, name)
        (out / f"{name}.error.json").write_text(json.dumps(vm.to_dict(e)))
    except Exception as e:  # noqa: BLE001 -- one size mustn't sink the run
        # Any other exception compiling or saving the size: that size's
        # CompileError, so the rest still compile.
        print(f"{name}: compile error")
        print(traceback.format_exc())
        _unsave(out, name)
        message = "".join(traceback.format_exception_only(e)).strip()
        (out / f"{name}.error.json").write_text(json.dumps({"CompileError": message}))


@beartype
def _unsave(out: Path, name: str) -> None:
    """
    Removes whatever of a size that failed was written, a part-saved `.vollo`
    included: a size with an error is never run, and its error is all
    `collect` reads.
    """
    for suffix in (".vollo", ".num-models", ".meta.json"):
        (out / f"{name}{suffix}").unlink(missing_ok=True)


@beartype
def compile_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    The `compile` command.
    """
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} compile", description="Compile the zoo for a hw config"
    )
    parser.add_argument("--config", required=True, help="the zoo's config name")
    parser.add_argument("--hw-config", required=True, help="hw config JSON")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--skip", action="append", default=[], help="<model> or <model>-<size>"
    )
    args = parser.parse_args(argv)
    if args.config not in vm.CONFIGS:
        parser.error(f"unknown --config {args.config}, one of {list(vm.CONFIGS)}")

    listing = catalog.listing()
    # A typo must not silently run a size the caller meant to exclude.
    known = {
        name
        for model, entry in listing.items()
        for name in (model, *(f"{model}-{size['name']}" for size in entry["sizes"]))
    }
    unknown = [skip for skip in args.skip if skip not in known]
    if unknown:
        parser.error(f"--skip names no model or size: {', '.join(unknown)}")

    hw_config = vc.Config.load(args.hw_config)
    # Not an error: a card's hw config may differ from the zoo's config.
    if not _same_hardware(hw_config, args.config):
        print(f"note: the hw config isn't the zoo's {args.config}")

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = []

    for model, entry in listing.items():
        named = catalog.named_sizes(model)
        for size in entry["sizes"]:
            name = size["name"]
            # The sizes the benchmarks sweep, of every model (experimental ones
            # included), but checked against the card's hw config, which may
            # lack a feature the zoo's config has.
            if not vm.in_sweep(named[name], hw_config) or _skipped(
                args.skip, model, name
            ):
                continue
            sizes.append(f"{model} {name}")
            print(f"Compiling {model}-{name}")
            _compile_size(model, name, args.config, hw_config, args.out)

    (args.out / "sizes.txt").write_text("".join(f"{s}\n" for s in sizes))

    return 0


@beartype
def config_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    The `config` command: prints the zoo config of a card.
    """
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} config", description="Print the zoo config of a card"
    )
    parser.add_argument("--hw-config", required=True, help="hw config JSON")
    args = parser.parse_args(argv)

    try:
        print(detect_config(vc.Config.load(args.hw_config)))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _read_json(path: Path):
    return json.loads(path.read_text())


@beartype
def _run_stats(run: dict) -> dict:
    # As floats: a C program may print a whole number as `3`, which `parse`'s
    # `float` annotations would reject.
    latency = run["metrics"]["latency_us"]
    median = latency.get("median")
    return {
        "mean": float(latency["mean"]),
        "median": float(median) if _is_number(median) else None,
        "p99": float(latency["p99"]),
    }


# What `_entry` reads of a run's `vollo-example --json` output. Every one must
# be a finite number > 0. scripts/fpga-benchmark.sh's `bench` checks the same keys.
_RUN_KEYS = [("metrics", "latency_us", "mean"), ("metrics", "latency_us", "p99")]


@beartype
def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_MISSING = object()


@beartype
def _get(run: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(run, dict) or key not in run:
            return _MISSING
        run = run[key]
    return run


@beartype
def _run_result(runs: Path, run: str, kind: str) -> dict:
    """
    The `vollo-example --json` output of one run, or `{"RunError": msg}` if it
    failed, or its output lacks a number `_entry` reads, or has one that isn't
    finite and positive: one odd output is that run's error, rather than
    failing the collection of every run.
    """
    error = runs / f"{run}.{kind}.run-error"
    if error.exists():
        return {"RunError": error.read_text().strip()}
    try:
        result = _read_json(runs / f"{run}.{kind}.json")
    except (OSError, ValueError):
        return {"RunError": "no run result"}
    values = {".".join(path): _get(result, path) for path in _RUN_KEYS}
    missing = [key for key, value in values.items() if not _is_number(value)]
    if missing:
        return {"RunError": f"vollo-example output lacks {', '.join(missing)}"}
    bad = [
        f"{key} = {value}"
        for key, value in values.items()
        if not (math.isfinite(value) and value > 0)
    ]
    if bad:
        return {"RunError": f"vollo-example output has {', '.join(bad)}"}
    return result


@beartype
def _measured(result: dict) -> bool:
    return "RunError" not in result


@beartype
def _entry(name: str, index: int, info: dict, runs: Path, config: str) -> dict:
    run = f"{name}.{index}"
    if (runs / f"{run}.skipped").exists():
        return {"Skipped": (runs / f"{run}.skipped").read_text().strip()}

    contiguous = _run_result(runs, run, "contiguous")
    spaced = _run_result(runs, run, "spaced")
    if not _measured(contiguous) and not _measured(spaced):
        return contiguous

    def stats(result: dict) -> dict:
        return _run_stats(result) if _measured(result) else result

    # The latency of a run that failed is null; its error is in `hardware`.
    return {
        "Ok": dict(
            config=config,
            param_count=info["param_count"],
            cycle_count=info["cycle_count"],
            latency_spaced=(
                {"microseconds": float(spaced["metrics"]["latency_us"]["mean"])}
                if _measured(spaced)
                else None
            ),
            # The unspaced run's mean latency, as the VM's `spaced=False` is:
            # one job at a time, so not 1 / its throughput.
            latency_contiguous=(
                {"microseconds": float(contiguous["metrics"]["latency_us"]["mean"])}
                if _measured(contiguous)
                else None
            ),
            meta=info["meta"],
            hardware=dict(
                name=name,
                model_index=index,
                contiguous=stats(contiguous),
                spaced=stats(spaced),
            ),
        )
    }


@beartype
def collect(programs: Path, runs: Path, config: str, version: str) -> dict:
    """
    The results JSON: `{version: {model: {config + FPGA_SUFFIX: [entry, ...]}}}`.
    """
    config += FPGA_SUFFIX
    results: dict = {}

    for line in (programs / "sizes.txt").read_text().splitlines():
        model, size = line.split()
        name = f"{model}-{size}"
        entries = results.setdefault(model, {config: []})[config]

        if (programs / f"{name}.error.json").exists():
            entries.append(_read_json(programs / f"{name}.error.json"))
        else:
            infos = _read_json(programs / f"{name}.meta.json")
            for index, info in enumerate(infos):
                entries.append(_entry(name, index, info, runs, config))

    return {version: results}


@beartype
def collect_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    The `collect` command.
    """
    parser = argparse.ArgumentParser(
        prog=f"{_PROG} collect", description="Assemble the hardware results JSON"
    )
    parser.add_argument("--config", required=True, help="the zoo's config name")
    parser.add_argument("--programs", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    data = collect(args.programs, args.runs, args.config, benchmark_version())
    args.out.write_text(json.dumps(data, indent=2))

    return 0


_PROG = "python -m vollo_model_zoo.fpga"

_COMMANDS = {"compile": compile_main, "collect": collect_main, "config": config_main}


@beartype
def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    `python -m vollo_model_zoo.fpga <command> [args]`, one of `_COMMANDS`.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG, description="The hardware path of scripts/fpga-benchmark.sh"
    )
    parser.add_argument("command", choices=list(_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args.args)


if __name__ == "__main__":
    sys.exit(main())
