import importlib
import os
from collections.abc import Generator
from dataclasses import asdict, dataclass
from functools import cache, partial

import numpy as np
import torch
import vollo_compiler as vc
import vollo_torch as vt
from beartype import beartype
from beartype.typing import Callable, Optional, Sequence, Union
from vollo_compiler import AllocationError, SaveError

# Names changed over time, fallback to depreciated for backwards-compat
_CONFIG_CONSTRUCTORS = {
    "V80": ("amd_v80_c6b32", "v80_c6b32"),
    "V80LL": ("amd_v80ll_c6b32", "v80ll_c6b32"),
    "IA-420f": ("bittware_ia420f_c6b32", "ia_420f_c6b32"),
    "IA-840f": ("bittware_ia840f_c3b64", "ia_840f_c3b64"),
    "NT400D11": ("napatech_nt400d11_c6b32", "nt400d11_c6b32"),
}


@beartype
def _get_configs() -> dict[str, vc.Config]:
    """
    Older vollo versions may not have all the configs.
    """
    configs = {}

    for name, constructors in _CONFIG_CONSTRUCTORS.items():
        for constructor in constructors:
            if hasattr(vc.Config, constructor):
                configs[name] = getattr(vc.Config, constructor)()
                break

    return configs


CONFIGS = _get_configs()


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
    """One entry point in a multi-model Vollo program."""

    name: str
    model: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    # One axis, or one per input for a multi-input model.
    streaming_axis: Optional[Union[int, tuple[int, ...]]] = None
    meta: Optional[dict[str, Union[int, float, str]]] = None


type Result = Union[Ok, AllocationError, SaveError, ValueError]


@beartype
def _config(conf: str) -> vc.Config:
    if conf not in CONFIGS:
        raise ValueError(
            f"Unknown config: {conf}, valid configs are: {list(CONFIGS.keys())}"
        )
    return CONFIGS[conf]


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
) -> Result:
    """
    For a given model/input compile it to a vollo program and return
    key information about the model and it's performance.
    """

    try:
        p = _vollo_compile(
            model,
            x,
            time_axis=time_axis,
            config=_config(config),
            allow_dynamic_weights=allow_dynamic_weights,
            quick_compile=quick_compile,
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
def vollo_multi_model_info(
    entries: Sequence[MultiModelEntry],
    *,
    config: str,
    quick_compile: bool = False,
) -> list[Result]:
    """Compile several PyTorch entry points into one Vollo program."""

    if not entries:
        raise ValueError("Expected at least one multi-model entry")

    try:
        builder = vc.ProgramBuilder(_config(config), remember_allocations=True)

        for entry in entries:
            shaped_model, _ = vt.fx.prepare_shape(entry.model, *entry.inputs)
            nnir = vt.fx.nnir.to_nnir(shaped_model)
            if entry.streaming_axis is not None:
                nnir, _ = nnir.streaming_transform(entry.streaming_axis)

            builder.add_nnir(nnir, entry.name, quick_compile=quick_compile)

        program = builder.to_program()
        program.pack()
    except (AllocationError, SaveError, ValueError) as error:
        return [error]

    # Shared modules, such as RNN-T's joint network, are counted once.
    parameters = {
        id(parameter): parameter
        for entry in entries
        for parameter in entry.model.parameters()
    }
    parameter_count = sum(parameter.numel() for parameter in parameters.values())

    results = []
    for model_index, entry in enumerate(entries):
        entry_meta = {"entry_point": entry.name, **(entry.meta or {})}
        results.append(
            Ok(
                config=config,
                param_count=parameter_count,
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
                meta=entry_meta,
            )
        )

    return results


type Activation = Callable[[torch.Tensor], torch.Tensor]


@cache
@beartype
def vollo_fn(fn: Activation, config: str) -> Activation:
    """
    Convert a torch activation function to a vollo activation function.

    When the resultant functions is used the following conversions occur:
        1. Torch tensor -> Numpy array
        2. Array scalar type -> np.float32 (since vollo VM only supports scalar float32)
        3. Vollo runs the function
        4. Vollo returns an fp32 which is then cast to a python scalar (fp64)
        5. These are aggregated back into an fp64 numpy array
        6. The numpy array is converted back to an fp64 torch tensor
    """

    # Dummy model for vollo to compile
    class Model(torch.nn.Module):
        def forward(self, x):
            return fn(x)

    # Use scalar effective input
    program = _vollo_compile(
        Model(),
        torch.tensor([1.0]),
        time_axis=None,
        config=_config(config),
        inputs_precisions=[vc.NumberFormat.FP32],
        outputs_precisions=[vc.NumberFormat.FP32],
    )

    return partial(
        _tensor_fn, fn=partial(_scalar_fn, vm=program.to_vm(bit_accurate=True))
    )


@np.vectorize
@beartype
def _scalar_fn[T](x: T, vm) -> T:
    """
    Run a scalar through the vollo VM.
    """
    return vm.run(np.atleast_1d(x).astype(np.float32)).item()


@beartype
def _tensor_fn(x: torch.Tensor, fn: Callable[[np.ndarray], np.ndarray]) -> torch.Tensor:
    """
    Performs: x -> .numpy() -> numpy_fn(x) -> from_numpy()
    """
    return torch.from_numpy(fn(x.numpy()))


@beartype
def _vollo_compile(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    time_axis: Optional[int],
    config: vc.Config,
    allow_dynamic_weights: bool = False,
    quick_compile: bool = False,
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
        config, quick_compile=quick_compile, allow_dynamic_weights=allow_dynamic_weights
    )

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
        raise ValueError(f"Unknown result type: {type(r)}")


@beartype
def get_models() -> list[str]:
    """
    Parse the models directory to find available models. Each model should be a
    .py file not starting with __ (to exclude __init__.py, etc.). The model
    name is derived from the filename.
    """
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    @beartype
    def is_valid_model_file(filename: str) -> bool:
        return filename.endswith(".py") and not filename.startswith("__")

    return sorted(f[:-3] for f in os.listdir(models_dir) if is_valid_model_file(f))


@beartype
def get_results(
    model_name: str, config: Optional[str]
) -> Generator[Result, None, None]:
    """
    Import the specified model module and call its main() function, this is
    expected to be a generator that yields Result objects.
    """
    model_module_path = f"vollo_model_zoo.models.{model_name}"
    model_module = importlib.import_module(model_module_path)

    if not hasattr(model_module, "main"):
        raise ImportError(
            f"Model module '{model_module_path}' does not have a callable main() function."
        )

    if config is None:
        return model_module.main()
    else:
        return model_module.main(config=config)
