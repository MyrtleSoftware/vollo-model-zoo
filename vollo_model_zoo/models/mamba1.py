import math
from pathlib import Path
from typing import Literal

import torch
import vollo_torch
from beartype import beartype
from beartype.typing import Generator
from torch import nn


class _MambaStep(nn.Module):
    @beartype
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int | Literal["auto"] = "auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        """
        Second half of mamba (post convolution) mainly the SSM.
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16) if isinstance(dt_rank, str) else dt_rank
        )

        # -- Customization

        self.exp = torch.exp
        self.act = torch.nn.Softplus()

        # -- Mamba derived weights

        # This is equal to `A_log.T` with A_log from mamba
        self.A_log_t = nn.Parameter(
            torch.arange(1, self.d_state + 1, dtype=torch.float32)
            .repeat(self.d_inner, 1)
            .t()
            .log()
            .contiguous()
        )

        # These are a single linear layer ("x_proj") in Mamba
        self.x_proj_t = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.x_proj_B = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.x_proj_C = nn.Linear(self.d_inner, self.d_state, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.D = nn.Parameter(torch.ones(1, self.d_inner))

        # -- Weight initialization

        # Without proper initialization, the (untrained) model will produce
        # outputs that explode in magnitude after a few steps.

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    @beartype
    def _init_state(self):
        return torch.zeros(
            self.d_state,
            self.d_model * self.expand,
        )

    def forward(self, x, state):
        """
        x     [d_inner         ]
        state [d_state, d_inner]
        """

        # This can be fused into a single mat-mul but I've left them
        # separate to mirror the reference implementation.
        # Compiler should fuse them if it's an optimisation
        dt = self.dt_proj(self.x_proj_t(x))

        B = self.x_proj_B(x)  # [s]
        C = self.x_proj_C(x)  # [s]

        dt = self.act(dt)  # [D]
        dA = self.exp(dt[None] * -self.A_log_t.exp())  # A_t = [s, D]

        # This is a trick to broadcast and change data dimension
        # Equivalent to: bcast_C = C[:, None] with rhs = data dimension
        bcast_C = torch.stack([C[i : i + 1] for i in range(self.d_state)])

        def sum(x):
            return x.sum(dim=0, keepdim=True)

        state = state * dA

        # Using a matmul `state.T @ C` is slower here.
        y = sum(state * bcast_C) + (x * dt) * sum(B * C)

        # === Now on the slow path ===

        # Same trick
        bcast_B = torch.stack([B[i : i + 1] for i in range(self.d_state)])

        # Core SSM step
        state = state + bcast_B * (x * dt)[None]

        # Inner residual
        y = y + self.D * x[None]

        return y.squeeze(), state


class Mamba(nn.Module):
    @beartype
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | Literal["auto"] = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        activation: Literal["silu", "relu"] = "silu",
    ):
        """
        See: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

        Args:
            d_model:    Dimension of the input and output.
            d_state:    SSM state expansion factor (dimension-1 of the rank-2 hidden state).
            d_conv:     Local convolution width.
            expand:     Hidden state expansion factor.
            dt_rank:    Generalized delta dimension.
            bias:       Input/output projection bias.
            conv_bias:  Convolutional bias.
            activation: Activation function to use for convolution/gate.
        """

        super().__init__()

        self.bias = bias

        step = _MambaStep(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            dt_rank=dt_rank,
        )

        self.ssm = vollo_torch.nn.Scan(step)

        # Not technically a parameter but needed for .to() etc
        self.h0 = torch.nn.Buffer(step._init_state(), persistent=False)

        # - Mamba parameters

        # These are a single linear layer ("in_proj") in Mamba
        self.in_proj_x = nn.Linear(step.d_model, step.d_inner, bias=bias)
        self.in_proj_z = nn.Linear(step.d_model, step.d_inner, bias=bias)

        # Depthwise
        self.conv1d = vollo_torch.nn.PaddedConv1d(
            in_channels=step.d_inner,
            out_channels=step.d_inner,
            groups=step.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
        )

        match activation:
            case "silu":
                self.act = nn.SiLU()
            case "relu":
                self.act = nn.ReLU()
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        self.out_proj = nn.Linear(step.d_inner, step.d_model, bias=bias)

    def forward(self, x):
        """
        x: [time, d_model]

        Returns r: [time, d_model]
        """

        # Up projection
        x, z = self.in_proj_x(x), self.in_proj_z(x)

        # Vollo requires that time is rightmost dimension for convolution
        x = x.transpose(0, 1)
        x = self.act(self.conv1d(x))
        x = x.transpose(0, 1)

        y = self.ssm(x, self.h0, input_axis=0, output_axis=0)  # [t, D]

        # Outer residual/gate
        y = y * self.act(z)

        # Down projection
        y = self.out_proj(y)  # [t, d]

        return y


@beartype
def _vm(
    dim: int,
    state: int,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, dim)

    model = Mamba(
        d_model=dim,
        d_state=state,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            state=state,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, state=6),
        dict(dim=32 * 12, state=6 * 2),
        dict(dim=32 * 32, state=6 * 2),
        dict(dim=32 * 32, state=6 * 4),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
