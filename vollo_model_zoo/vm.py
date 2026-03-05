from dataclasses import dataclass
from functools import cache, partial
from typing import Callable, Optional, Union

import numpy as np
import torch
import vollo_compiler as vc
import vollo_torch as vt
from beartype import beartype
from vollo_compiler import AllocationError, SaveError

CONFIGS = {
    "V80": vc.Config.v80_c6b32(),
    "V80LL": vc.Config.v80ll_c6b32(),
    "IA-420f": vc.Config.ia_420f_c6b32(),
    "IA-840f": vc.Config.ia_840f_c3b64(),  # TODO: or ia_840f_c2b64d()?
    "NT400D11": vc.Config.nt400d11_c6b32(),
}


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


type Result = Union[Ok, AllocationError, SaveError]


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
        )
    except (AllocationError, SaveError) as e:
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


type Activation = Callable[[torch.Tensor], torch.Tensor]


@cache
@beartype
def vollo_fn(fn: Activation, config: str) -> Activation:
    """
    Convert a torch activation function to a vollo activation function.

    When the resultant functions is used the following conversions occur:
        1. Torch tensor -> Numpy array
        2. Array scalar type -> np.float32 (since vollo VM only supports scalar float32)
        3. Vollo converts the float32 scalar to bf16 then runs the function
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
        Model(), torch.tensor([0.0]), time_axis=None, config=_config(config)
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
) -> vc.Program:
    # This gives nicer error messages as a first pass.
    model(x)

    model, _ = vt.fx.prepare_shape(model, x)

    nnir = vt.fx.nnir.to_nnir(model)

    if time_axis is not None:
        nnir, _ = nnir.streaming_transform(time_axis)

    program = nnir.to_program(config, allow_dynamic_weights=allow_dynamic_weights)

    program.pack()  # Should raise error if it doesn't fit

    return program
