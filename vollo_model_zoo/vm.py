import copy
import importlib
from collections.abc import Generator
from dataclasses import asdict, dataclass

import torch
import vollo_compiler as vc
import vollo_torch as vt
from beartype import beartype
from beartype.typing import Optional, Sequence, Union
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo import catalog

CONFIGS = {
    "V80": vc.Config.amd_v80_c6b32(),
    "V80LL": vc.Config.amd_v80ll_c6b32(),
    "IA-420f": vc.Config.bittware_ia420f_c6b32(),
    "IA-840f": vc.Config.bittware_ia840f_c3b64(),
    "NT400D11": vc.Config.napatech_nt400d11_c6b32(),
}


# Whether `sweep` includes the sizes marked `_experimental`. Only
# `zoo --experimental` and the tests set it, so benchmarks leave them out.
sweep_experimental = False

# The config `zoo` runs when `--config` is not given.
DEFAULT_CONFIG = "V80"


@beartype
@dataclass(frozen=True)
class Microseconds:
    """
    Type safe wrapper for microseconds.
    """

    microseconds: float

    @beartype
    def __repr__(self) -> str:
        return f"{self.microseconds}us"


@beartype
@dataclass
class Ok:
    """
    Generic VM result object.

    Members:
        param_count: Number of parameters in the model.
        cycle_count: Number of cycles per inference.
        latency_spaced: Spaced inference compute latency.
        latency_contiguous: Back-to-back inference compute latency.

        meta: Other model metadata, i.e. layers, hidden size, etc.
              use `_` prefixed-keys to hide from default output.
    """

    config: str
    param_count: int
    cycle_count: int
    latency_spaced: Microseconds
    latency_contiguous: Microseconds

    meta: Optional[dict[str, Union[int, float, str]]] = None


@beartype
@dataclass(frozen=True)
class MultiModelEntry:
    """
    One entry point in a multi-model Vollo program.
    """

    name: str
    model: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    # One axis, or one per input for a multi-input model.
    streaming_axis: Optional[Union[int, tuple[int, ...]]] = None
    meta: Optional[dict[str, Union[int, float, str]]] = None


type Result = Union[Ok, AllocationError, SaveError, ValueError]


@beartype
def _config(conf: str, boards: int = 1) -> vc.Config:
    """
    The config `conf`, or for `boards` > 1, that many boards' worth of its
    cores as one config.
    """
    if conf not in CONFIGS:
        raise ValueError(
            f"Unknown config: {conf}, valid configs are: {list(CONFIGS.keys())}"
        )
    if boards == 1:
        return CONFIGS[conf]
    config = copy.copy(CONFIGS[conf])
    config.num_cores *= boards
    return config


@beartype
def config_supports(config: str, feature: str) -> bool:
    """
    Whether a config's fabric provides a compiler feature, e.g. `"fp8"`.
    """
    return feature in _config(config).features


@beartype
def runs_on(size: dict, config: str) -> bool:
    """
    Whether `config` provides the compiler features a size requires (its
    `_features`, see `catalog.split`).
    """
    _, requirements = catalog.split(size)
    return _provides(config, requirements)


@beartype
def _provides(config: str, requirements: dict) -> bool:
    _config(config)
    features = requirements.get("_features", [])
    return all(config_supports(config, feature) for feature in features)


@beartype
def vollo_info(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    time_axis: Optional[int],
    config: str,
    meta: Optional[dict[str, Union[int, float, str]]] = None,
    allow_dynamic_weights: bool = False,
    quick_compile: bool = False,
    allow_unserializable: bool = False,
    boards: int = 1,
) -> Result:
    """
    For a given model/input compile it to a vollo program and return
    key information about the model and it's performance.

    With `boards` > 1, the program is compiled for that many boards' worth of
    `config`'s cores, a stand-in for running across several boards. Above one
    board's worth of cores, a program can't be serialized, so this needs
    `allow_unserializable=True`.
    """

    try:
        p = _vollo_compile(
            model,
            x,
            time_axis=time_axis,
            config=_config(config, boards),
            allow_dynamic_weights=allow_dynamic_weights,
            quick_compile=quick_compile,
            allow_unserializable=allow_unserializable,
        )
    except (AllocationError, SaveError, ValueError) as e:
        return e

    latency_fast = p.compute_duration_per_inference_us(spaced=True)
    latency_slow = p.compute_duration_per_inference_us(spaced=False)

    return Ok(
        config=config,
        param_count=sum(p.numel() for p in model.parameters()),
        cycle_count=p.cycle_count_per_inference(),
        latency_spaced=Microseconds(latency_fast),
        latency_contiguous=Microseconds(latency_slow),
        meta=meta,
    )


@beartype
def _compile_entries(
    entries: Sequence[MultiModelEntry], config: vc.Config, quick_compile: bool = False
) -> vc.Program:
    """
    The one program of several entry points, packed (so it raises if it doesn't fit).
    """
    builder = vc.ProgramBuilder(config, remember_allocations=True)

    for entry in entries:
        shaped_model, _ = vt.fx.prepare_shape(entry.model, *entry.inputs)
        nnir = vt.fx.nnir.to_nnir(shaped_model)
        if entry.streaming_axis is not None:
            nnir, _ = nnir.streaming_transform(entry.streaming_axis)

        builder.add_nnir(nnir, entry.name, quick_compile=quick_compile)

    program = builder.to_program()
    program.pack()
    return program


@beartype
def compile_instance(zoo: dict, config: vc.Config) -> vc.Program:
    """
    The program of a size, as `catalog.build` returns it, for `config`: the
    compile `vollo_info` / `vollo_multi_model_info` do, but handing back the
    program and raising what they return, so a caller can e.g. save it
    (`fpga.py`).
    """
    if "entries" in zoo:
        unsupported = zoo.keys() - {"entries", "quick_compile"}
        assert not unsupported, f"Unsupported zoo arguments: {sorted(unsupported)}"
        entries = [MultiModelEntry(**entry) for entry in zoo["entries"]]
        return _compile_entries(entries, config, zoo.get("quick_compile", False))

    supported = {
        "model",
        "x",
        "time_axis",
        "meta",
        "allow_dynamic_weights",
        "quick_compile",
        "allow_unserializable",
        "boards",
    }
    unsupported = zoo.keys() - supported
    assert not unsupported, f"Unsupported zoo arguments: {sorted(unsupported)}"
    # `config` is one board's; compiling a multi-board size for it would
    # silently measure something else.
    assert zoo.get("boards", 1) == 1, f"Can't compile for {zoo['boards']} boards"
    return _vollo_compile(
        zoo["model"],
        zoo["x"],
        time_axis=zoo["time_axis"],
        config=config,
        allow_dynamic_weights=zoo.get("allow_dynamic_weights", False),
        quick_compile=zoo.get("quick_compile", False),
        allow_unserializable=zoo.get("allow_unserializable", False),
    )


@beartype
def vollo_multi_model_info(
    entries: Sequence[dict],
    *,
    config: str,
    quick_compile: bool = False,
) -> list[Result]:
    """
    Compile several PyTorch entry points into one Vollo program. Each entry is
    a plain dict of `MultiModelEntry`'s fields, which is what model files
    return so they need not import the zoo.
    """

    if not entries:
        raise ValueError("Expected at least one multi-model entry")
    entries = [MultiModelEntry(**entry) for entry in entries]

    try:
        program = _compile_entries(entries, _config(config), quick_compile)
    except (AllocationError, SaveError, ValueError) as error:
        return [error]

    results = []
    for model_index, entry in enumerate(entries):
        results.append(
            Ok(
                config=config,
                # Each entry's own, so a module shared between entries counts in each.
                param_count=sum(p.numel() for p in entry.model.parameters()),
                cycle_count=program.cycle_count_per_inference(model_index=model_index),
                latency_spaced=Microseconds(
                    program.compute_duration_per_inference_us(
                        model_index=model_index,
                        spaced=True,
                    )
                ),
                latency_contiguous=Microseconds(
                    program.compute_duration_per_inference_us(
                        model_index=model_index,
                        spaced=False,
                    )
                ),
                meta={"prog": entry.name, **(entry.meta or {})},
            )
        )

    return results


@beartype
def _vollo_compile(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    time_axis: Optional[int],
    config: vc.Config,
    allow_dynamic_weights: bool = False,
    quick_compile: bool = False,
    allow_unserializable: bool = False,
    **kwargs,
) -> vc.Program:
    """
    kwargs are forwarded to vt.fx.nnir.to_nnir, e.g. inputs_precisions, outputs_precisions, etc.
    """
    # This gives nicer error messages as a first pass.
    model(x)

    model, _ = vt.fx.prepare_shape(model, x)

    nnir = vt.fx.nnir.to_nnir(model, **kwargs)

    if time_axis is not None:
        nnir, _ = nnir.streaming_transform(time_axis)

    program = nnir.to_program(
        config,
        quick_compile=quick_compile,
        allow_dynamic_weights=allow_dynamic_weights,
        allow_unserializable=allow_unserializable,
    )
    if not allow_unserializable:
        program.pack()  # Should raise error if it doesn't fit

    return program


@beartype
def to_dict(r: Result) -> dict:
    if isinstance(r, Ok):
        return {"Ok": asdict(r)}
    elif isinstance(r, AllocationError):
        return {"AllocationError": 0}
    elif isinstance(r, SaveError):
        return {"SaveError": 1}
    elif isinstance(r, ValueError):
        return {"ValueError": str(r)}
    else:
        raise TypeError(f"Unknown result type: {type(r)}")


@beartype
def sweep(sizes: Sequence[dict], config: str) -> Generator[dict]:
    """
    The sizes in a model's `SIZES` that `config` can run (see `runs_on`), in
    order, as arguments to the model's `instance`. Sizes marked `_experimental`
    are left out unless `sweep_experimental` is set.
    """
    for size in sizes:
        arguments, requirements = catalog.split(size)
        if requirements.get("_experimental") and not sweep_experimental:
            continue
        if _provides(config, requirements):
            yield arguments


@beartype
def get_results(model_name: str, config: str) -> Generator[Result]:
    """
    Compile each size of the zoo model `model_name`'s `SIZES` that `config` can
    run (see `sweep`), as built by its `instance`, and yield the results in
    order.
    """
    module = importlib.import_module(f"vollo_model_zoo.models.{model_name}")
    for size in sweep(module.SIZES, config):
        built = module.instance(**size, config=_config(config))
        if "entries" in built:
            yield from vollo_multi_model_info(**built, config=config)
        else:
            yield vollo_info(**built, config=config)
