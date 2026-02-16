from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import vollo_compiler as vc
import vollo_torch as vt
from beartype import beartype


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
class Result:
    """
    Generic VM result object.

    Members:
        param_count: Number of parameters in the model.
        cycle_count: Number of cycles per inference.
        latency_fast: Spaced inference compute latency.
        latency_slow: Back-to-back inference compute latency.

        layers: Number of layers (or-equivilent) in the model.
        hidden: Hidden-size (if applicable) of the model.

        meta: Other model metadata.
    """

    param_count: int
    cycle_count: int
    latency_fast: Microseconds
    latency_slow: Microseconds

    layers: Optional[int] = None
    hidden: Optional[int] = None

    meta: Optional[dict[str, Union[int, float, str]]] = None


_DEFAULT_CONFIG: vc.Config = vc.Config.v80_c6b32()

# TODO: what happens when it's not a streaming model?


@beartype
def _vollo_compile(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    time_axis: Optional[int],
    config: vc.Config,
) -> vc.Program:
    # This gives nicer error messages as a first pass.
    model(x)

    model, _ = vt.fx.prepare_shape(model, x)

    nnir = vt.fx.nnir.to_nnir(model)

    if time_axis is not None:
        nnir, _ = nnir.streaming_transform(time_axis)

    program = nnir.to_program(config)

    program.pack()  # Should raise error if it doesn't fit

    return program


@beartype
def vollo_info(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    layers: Optional[int],
    hidden: Optional[int],
    time_axis: Optional[int],
    meta: Optional[dict[str, Union[int, float, str]]] = None,
    config: vc.Config = _DEFAULT_CONFIG,
) -> Result:
    """
    For a given model/input compile it to a vollo program and return
    key information about the model and it's performance.
    """
    p = _vollo_compile(model, x, time_axis=time_axis, config=config)

    latency_fast = p.compute_duration_per_inference_us(spaced=True)
    latency_slow = p.compute_duration_per_inference_us(spaced=False)

    return Result(
        param_count=sum(p.numel() for p in model.parameters()),
        cycle_count=p.cycle_count_per_inference(),
        latency_fast=Microseconds(latency_fast),
        latency_slow=Microseconds(latency_slow),
        layers=layers,
        hidden=hidden,
        meta=meta,
    )
