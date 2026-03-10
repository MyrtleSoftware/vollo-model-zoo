from contextlib import nullcontext
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
        d_state: int = 32,
        d_conv: int = 4,
        d_head: int = 32,
        expand: int | float = 2,
        bias: bool = False,
        conv_bias: bool = True,
        activation: Literal["silu", "relu"] = "silu",
        ssm_fp32: bool = True,
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
            ssm_fp32:   Whether to use fp32 for the ssm hidden state and select activations.
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

        self.norm = torch.nn.RMSNorm(self.d_inner, eps=1e-5)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # === Vollo specific === #

        self.ssm = vollo_torch.nn.Scan(
            _Mamba2Step(
                n_heads=self.n_heads, d_head=self.d_head, d_state=d_state, fp32=ssm_fp32
            )
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
        dA = dt * (-torch.exp(self.A_log))

        x_reshaped = x.reshape(-1, self.n_heads, self.d_head)

        y = self.ssm(
            [x_reshaped, B, C, dt, dA], self.h0, input_axis=[0] * 5, output_axis=0
        )

        # [t h! p] -> [t (h p)!]
        y = y.reshape(-1, self.d_inner)

        # Skip connection
        y = y + self.D * x

        # Gating (FLA always uses SiLU regardless of hidden_act)
        # This is "norm after gate" configuration.
        y = y * torch.nn.functional.silu(z)

        y = self.norm(y)

        return self.out_proj(y)


class _Mamba2Step(nn.Module):
    @beartype
    def __init__(self, n_heads: int, d_head: int, d_state: int, fp32: bool):
        super().__init__()

        self.n_heads = n_heads  # h
        self.d_head = d_head  # p
        self.d_state = d_state  # n

        self.fp_context = vollo_torch.Fp32Activations if fp32 else nullcontext

    def forward(self, inputs: list[torch.Tensor], h: torch.Tensor):
        """
        Inputs:
                 x: [h p!]
              B, C: [n!]
            dA, dt: [h!]
                 h: [h p! n]

        Returns:
                 y: [h p!]
        """
        x, B, C, dt, dA = inputs

        with self.fp_context():
            dA = dA.exp()

        # dA [h!] -> [h 1!]
        dA = torch.stack([dA[i : i + 1] for i in range(self.n_heads)], dim=0)

        # dt [h!] -> [h 1!]
        dt = torch.stack([dt[i : i + 1] for i in range(self.n_heads)], dim=0)

        # [h p! n] @ [n!] -> [h p!]
        y = dA * (h @ C) + dt * x * (B * C).sum(0, keepdim=True)

        # [h p! 1] * [1 n 1!]
        dB = (dt * x)[:, :, None] * torch.stack(
            [B[i : i + 1] for i in range(self.d_state)], dim=1
        )

        with self.fp_context():
            h = dA[:, :, None] * h + dB

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
    fp32: bool,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(2, dim)

    model = nn.Sequential().extend(
        Mamba2(d_model=dim, d_state=state, ssm_fp32=fp32) for _ in range(layers)
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            fp32=fp32,
            dim=dim,
            state=state,
            layers=layers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=400, state=16, layers=1, fp32=False),
        dict(dim=400, state=16, layers=1, fp32=True),
        dict(dim=400, state=32, layers=1, fp32=False),
        dict(dim=400, state=32, layers=1, fp32=True),
        dict(dim=400, state=16, layers=2, fp32=False),
        dict(dim=400, state=16, layers=2, fp32=True),
        dict(dim=1024, state=32, layers=1, fp32=False),
        dict(dim=1024, state=32, layers=1, fp32=True),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
