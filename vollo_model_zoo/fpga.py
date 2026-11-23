"""
The hardware path of the zoo, driven by `scripts/fpga-benchmark.sh`:

- `fpga-compile` compiles every size to a `.vollo` program for a hw config read
  from a card, using the public compiler API only (so it works with released
  SDKs). This is the one place the zoo saves programs; `vm.py` never does.
- `fpga-collect` turns the compile outputs and the `vollo-example` runs into a
  results JSON in the shape of `benchmarks/*.json`.

Files in the programs directory, per size `<model>-<size>`:
    .vollo + .num-models + .meta.json   compiled
    .no-fit                             AllocationError / SaveError
    .compile-error                      any other exception, with traceback
and `sizes.txt` lists `<model> <size>` lines. Files in the runs directory, per
model index `<model>-<size>.<index>`: `.raw.json`, `.spaced.json` (the
`vollo-example --json` output), `.raw.run-error` / `.spaced.run-error` (a run
of that kind failed) or `.skipped` (a message each).
An index with one run failed is still `Ok`: that run's latency is null and its
side of `hardware` is `{"RunError": msg}`. Only both failing is a `RunError`.
Sizes excluded by --skip are left out entirely.

Note: latencies here are measured on the card, so unlike the VM's they include
host<->accelerator IO.
"""

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import vollo_compiler as vc
from beartype import beartype
from beartype.typing import Optional, Sequence

from vollo_model_zoo import catalog, vm
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

    Configs are first matched on hardware alone, as an older release's name and
    clock may differ from the zoo's. Some zoo configs are the same hardware
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
def _model_info(zoo: dict, program: vc.Program) -> list[dict]:
    """
    Per model of the program: its parameter count, cycles and meta.
    """

    def cycles(index: int) -> int:
        if "entries" in zoo:
            return program.cycle_count_per_inference(model_index=index)
        return program.cycle_count_per_inference()

    if "entries" in zoo:
        # Each entry's own, so a module shared between entries counts in each,
        # as `vm.vollo_multi_model_info` counts them.
        return [
            dict(
                param_count=sum(p.numel() for p in entry["model"].parameters()),
                cycle_count=cycles(i),
                meta={"prog": entry["name"], **(entry.get("meta") or {})},
            )
            for i, entry in enumerate(zoo["entries"])
        ]

    count = sum(p.numel() for p in zoo["model"].parameters())
    return [dict(param_count=count, cycle_count=cycles(0), meta=zoo.get("meta"))]


@beartype
def _skipped(skips: Sequence[str], model: str, size: str) -> bool:
    return any(skip in (model, f"{model}-{size}") for skip in skips)


@beartype
def _compile_size(
    model: str, size: str, config: str, hw_config: vc.Config, out: Path
) -> None:
    name = f"{model}-{size}"
    try:
        zoo = catalog.build(model, size, config)
        program = vm.compile_instance(zoo, hw_config)
        info = _model_info(zoo, program)
        program.save(str(out / f"{name}.vollo"))
        # Old SDKs have single-model programs only.
        num_models = program.num_models() if hasattr(program, "num_models") else 1
        (out / f"{name}.num-models").write_text(f"{num_models}\n")
        (out / f"{name}.meta.json").write_text(json.dumps(info))
    except (vc.AllocationError, vc.SaveError) as e:
        print(f"{name}: doesn't fit: {e}")
        (out / f"{name}.no-fit").write_text(
            "SaveError\n" if isinstance(e, vc.SaveError) else "AllocationError\n"
        )
    except AssertionError:
        # `compile_instance` rejecting a zoo argument is a bug in the zoo, not
        # something an SDK did, so it must not pass for a CompileError.
        raise
    except Exception:
        # Old SDKs fail in ways the current one doesn't.
        print(f"{name}: compile error")
        error = traceback.format_exc()
        print(error)
        (out / f"{name}.compile-error").write_text(error)


@beartype
def compile_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the 'fpga-compile' command.
    """
    parser = argparse.ArgumentParser(description="Compile the zoo for a hw config")
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
    # Not an error: an older release's hardware may differ from the zoo's config.
    if not _same_hardware(hw_config, args.config):
        print(f"note: the hw config isn't the zoo's {args.config}")

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = []

    for model, entry in listing.items():
        named = catalog.named_sizes(model)
        for size in entry["sizes"]:
            # As the benchmarks: no experimental sizes, none the config can't run.
            if size["experimental"] is not None:
                continue
            if not vm.runs_on(named[size["name"]], args.config):
                continue

            name = size["name"]
            if _skipped(args.skip, model, name):
                continue
            sizes.append(f"{model} {name}")
            print(f"Compiling {model}-{name}")
            _compile_size(model, name, args.config, hw_config, args.out)

    (args.out / "sizes.txt").write_text("".join(f"{s}\n" for s in sizes))

    return 0


@beartype
def hw_config_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the 'fpga-hw-config' command: reads the output of
    `vollo-tool read-hw-config` on stdin and writes the first card's hw config.
    """
    parser = argparse.ArgumentParser(description="Extract a card's hw config")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    cards = json.load(sys.stdin)
    hw_config = cards[0].get("hw_config") if cards else None
    if hw_config is None:
        print("error: the device has no hw config", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(hw_config))
    return 0


@beartype
def config_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the 'fpga-config' command: prints the zoo config of a card.
    """
    parser = argparse.ArgumentParser(description="Print the zoo config of a card")
    parser.add_argument("--hw-config", required=True, help="hw config JSON")
    args = parser.parse_args(argv)

    try:
        print(detect_config(vc.Config.load(args.hw_config)))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


@beartype
def check_json_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the 'fpga-check-json' command: 0 if a file is valid JSON.
    """
    parser = argparse.ArgumentParser(description="Check a file is valid JSON")
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)

    try:
        json.loads(args.file.read_text())
    except (OSError, ValueError):
        return 1
    return 0


def _read_json(path: Path):
    return json.loads(path.read_text())


@beartype
def _run_stats(run: dict) -> dict:
    latency = run["metrics"]["latency_us"]
    return {
        "mean": latency["mean"],
        "median": latency.get("median"),
        "p99": latency["p99"],
    }


@beartype
def _run_result(runs: Path, run: str, kind: str) -> dict | str:
    """
    The `vollo-example --json` output of one run, or its error if it failed.
    """
    error = runs / f"{run}.{kind}.run-error"
    if error.exists():
        return error.read_text().strip()
    try:
        return _read_json(runs / f"{run}.{kind}.json")
    except (OSError, ValueError):
        return "no run result"


@beartype
def _entry(
    name: str, index: int, info: dict, runs: Path, config: str, hw: dict
) -> dict:
    run = f"{name}.{index}"
    if (runs / f"{run}.skipped").exists():
        return {"Skipped": (runs / f"{run}.skipped").read_text().strip()}

    raw = _run_result(runs, run, "raw")
    spaced = _run_result(runs, run, "spaced")
    # Only a run that failed has a str, its error.
    if isinstance(raw, str) and isinstance(spaced, str):
        return {"RunError": raw}

    def stats(result) -> dict:
        return {"RunError": result} if isinstance(result, str) else _run_stats(result)

    # The latency of a run that failed is null; its error is in `hardware`.
    return {
        "Ok": dict(
            config=config,
            param_count=info["param_count"],
            cycle_count=info["cycle_count"],
            latency_spaced=(
                None
                if isinstance(spaced, str)
                else {"microseconds": spaced["metrics"]["latency_us"]["mean"]}
            ),
            latency_contiguous=(
                None
                if isinstance(raw, str)
                else {"microseconds": 1e6 / raw["metrics"]["throughput"]}
            ),
            meta=info["meta"],
            hardware=dict(
                name=name,
                model_index=index,
                raw_buffer=stats(raw),
                spaced=stats(spaced),
                hw_config=hw,
            ),
        )
    }


@beartype
def collect(
    programs: Path, runs: Path, config: str, hw_config: Path, version: str
) -> dict:
    """
    The results JSON: `{version: {model: {config: [entry, ...]}}}`.
    """
    hw = _read_json(hw_config)
    results: dict = {}

    for line in (programs / "sizes.txt").read_text().splitlines():
        model, size = line.split()
        name = f"{model}-{size}"
        entries = results.setdefault(model, {config: []})[config]

        if (programs / f"{name}.no-fit").exists():
            kind = (programs / f"{name}.no-fit").read_text().strip()
            # As `vm.to_dict` writes them.
            entries.append(
                {"SaveError": 1} if kind == "SaveError" else {"AllocationError": 0}
            )
        elif (programs / f"{name}.compile-error").exists():
            error = (programs / f"{name}.compile-error").read_text().strip()
            entries.append({"CompileError": error.splitlines()[-1] if error else ""})
        else:
            infos = _read_json(programs / f"{name}.meta.json")
            for index, info in enumerate(infos):
                entries.append(_entry(name, index, info, runs, config, hw))

    return {version: results}


@beartype
def collect_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the 'fpga-collect' command.
    """
    parser = argparse.ArgumentParser(description="Assemble the hardware results JSON")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hw-config", required=True, type=Path)
    parser.add_argument("--programs", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    data = collect(
        args.programs, args.runs, args.config, args.hw_config, benchmark_version()
    )
    args.out.write_text(json.dumps(data, indent=2))

    return 0
