import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import vollo_torch
from beartype import beartype
from beartype.typing import Generator
from torch import nn


class Mamba2(nn.Module):
    @beartype
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        d_head: int = 64,
        expand: int | float = 2,
        bias: bool = False,
        conv_bias: bool = True,
        activation: Literal["silu", "relu"] = "silu",
        rmsnorm: bool = True,
    ):
        """
        See: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

        Args:
            d_model:    Dimension of the input and output.
            d_state:    Dimension of the SSM state.
            d_conv:     Local convolution width.
            d_head:     Head dimension for multi-head projection.
            expand:     Hidden state expansion factor.
            bias:       Input/output projection bias.
            conv_bias:  Convolutional bias.
            activation: Activation function to use for convolution/gate.
        """
        super().__init__()

        self.d_inner = int(expand * d_model)
        self.d_head = d_head

        self.proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
        self.proj_x = nn.Linear(d_model, self.d_inner, bias=bias)

        self.proj_B = nn.Linear(d_model, d_state, bias=bias)
        self.proj_C = nn.Linear(d_model, d_state, bias=bias)

        assert self.d_inner % self.d_head == 0, "d_inner must be divisible by headdim"

        self.n_heads = self.d_inner // self.d_head

        self.proj_dt = nn.Linear(d_model, self.n_heads, bias=bias)

        self.conv_x = _ShortConv(
            dim=self.d_inner, d_conv=d_conv, bias=conv_bias, activation=activation
        )
        self.conv_B = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )
        self.conv_C = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )

        self.dt_bias = nn.Parameter(torch.rand(self.n_heads))
        self.A_log = nn.Parameter(torch.rand(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # if rmsnorm:
        #     self.norm = None  # d_inner

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # === Vollo specific === #

        self.ssm = vollo_torch.nn.Scan(
            _Mamba2Step(n_heads=self.n_heads, d_head=self.d_head, d_state=d_state)
        )

        self.h0 = torch.nn.Buffer(
            torch.zeros(self.n_heads, self.d_head, d_state), persistent=False
        )

    def forward(self, input):
        """
        Input:
            x: [T, D]
        Output:
            y: [T, D]
        """
        z = self.proj_z(input)
        x = self.proj_x(input)
        B = self.proj_B(input)
        C = self.proj_C(input)
        dt = self.proj_dt(input)

        x = self.conv_x(x)
        B = self.conv_B(B)
        C = self.conv_C(C)

        dt = F.softplus(dt + self.dt_bias)
        dA = dt

        if True:
            state = self.h0.float()
        else:
            state = self.h0

        # TODO: D/skip connections

        # x: [t (h p)] -> (t h p)

        x = x.reshape(-1, self.n_heads, self.d_head)

        y = self.ssm([x, B, C, dt, dA], state, input_axis=[0] * 5, output_axis=0)

        # [t h! p]

        y = y.reshape(-1, self.d_inner)

        return self.out_proj(y)


class _Mamba2Step(nn.Module):
    @beartype
    def __init__(self, n_heads: int, d_head: int, d_state: int):
        super().__init__()

        self.n_heads = n_heads  # h
        self.d_head = d_head  # p
        self.d_state = d_state  # n

        # print(f"h={n_heads}, p={d_head}, n={d_state}")

    def forward(self, inputs: list[torch.Tensor], h: torch.Tensor):
        """
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.rmsnorm:
            y = y * self.act(z)  # (B D)
        """

        # Shapes:
        #
        #   x:  [h p!]
        #   h:  [h p! n]
        #
        #   B:  [n!]
        #   C:  [n!]
        #
        #   dt: [h!]
        #   dA: [h!]
        #

        x, B, C, dt, dA = inputs

        dA = dA.exp()

        # dA [h!] -> [h 1!]
        dA = torch.stack([dA[i : i + 1] for i in range(self.n_heads)], dim=0)

        # dt [h!] -> [h 1!]
        dt = torch.stack([dt[i : i + 1] for i in range(self.n_heads)], dim=0)

        # [h p! n] @ [n!] -> h p!
        y = dA * (h @ C) + dt * x * (B * C).sum(0, keepdim=True)

        # We want: [h 1! 1] * [h p! 1] * [1 1! n]
        C = torch.stack([C[i : i + 1] for i in range(self.d_state)], dim=1)

        h = dA[:, :, None] * h + (dt * x)[:, :, None] * C

        return y, h


class _ShortConv(nn.Module):
    @beartype
    def __init__(
        self, dim: int, d_conv: int, bias: bool, activation: Literal["silu", "relu"]
    ):
        super().__init__()

        self.conv1d = vollo_torch.nn.PaddedConv1d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            kernel_size=d_conv,
            bias=bias,
        )

        match activation:
            case "silu":
                self.act = nn.SiLU()
            case "relu":
                self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolution: [T, D] -> [T, D]
        """
        x = x.transpose(0, 1)
        x = self.conv1d(x)
        x = self.act(x)
        x = x.transpose(0, 1)
        return x


@beartype
def _vm(
    dim: int,
    state: int,
    layers: int,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(2, dim)

    model = nn.Sequential().extend(
        Mamba2(d_model=dim, d_state=state) for _ in range(layers)
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
            layers=layers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6 * 2, state=32, layers=1),
        # dict(dim=32 * 6 * 2, state=24, layers=2),
        # dict(dim=32 * 6 * 4, state=24, layers=3),
        # dict(dim=32 * 12, state=128),
        # dict(dim=32 * 32, state=128),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
