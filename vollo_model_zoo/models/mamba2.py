from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional

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
        head_partitions: Optional[tuple[tuple[int, ...], ...]] = (
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
        ),
        headwise_linear: bool = True,
    ):
        """
        See: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

        Args:
            d_model:          Dimension of the input and output.
            d_state:          Dimension of the SSM state.
            d_conv:           Local convolution width.
            d_head:           Head dimension for multi-head projection.
            expand:           Hidden state expansion factor.
            bias:             Input/output projection bias.
            conv_bias:        Convolutional bias.
            activation:       Activation function to use for convolution/gate.
            ssm_fp32:         Whether to use fp32 for the ssm hidden state and select activations.
            head_partitions:  Sequence of core partition indices, one per head partition.
            headwise_linear:  Whether to run linear projections per head partition inside core partitions.
        """
        super().__init__()

        self.d_inner = int(expand * d_model)
        self.d_head = d_head
        assert self.d_inner % self.d_head == 0, "d_inner must be divisible by headdim"
        self.n_heads = self.d_inner // self.d_head

        if head_partitions is not None:
            num_partitions = len(head_partitions)
            if self.n_heads < num_partitions:
                head_partitions = None

        self.head_partitions = head_partitions
        self.headwise_linear = self.head_partitions is not None and headwise_linear

        self.proj_B = nn.Linear(d_model, d_state, bias=bias)
        self.proj_C = nn.Linear(d_model, d_state, bias=bias)

        self.conv_B = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )
        self.conv_C = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )

        if self.head_partitions is None:
            self.proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
            self.proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
            self.proj_dt = nn.Linear(d_model, self.n_heads, bias=bias)

            self.conv_x = _ShortConv(
                dim=self.d_inner, d_conv=d_conv, bias=conv_bias, activation=activation
            )

            self.dt_bias = nn.Parameter(torch.rand(self.n_heads))
            self.A_log = nn.Parameter(torch.rand(self.n_heads))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            self.ssm = vollo_torch.nn.Scan(
                _Mamba2Step(
                    n_heads=self.n_heads,
                    d_head=self.d_head,
                    d_state=d_state,
                    fp32=ssm_fp32,
                )
            )

            self.h0 = torch.nn.Buffer(
                torch.zeros(self.n_heads, self.d_head, d_state), persistent=False
            )
        else:
            num_partitions = len(self.head_partitions)
            self.head_splits = [
                len(chunk)
                for chunk in torch.arange(self.n_heads).tensor_split(num_partitions)
            ]
            self.dim_splits = [h_cnt * self.d_head for h_cnt in self.head_splits]

            self.ssms = nn.ModuleList(
                [
                    vollo_torch.nn.Scan(
                        _Mamba2Step(
                            n_heads=self.head_splits[p],
                            d_head=self.d_head,
                            d_state=d_state,
                            fp32=ssm_fp32,
                        )
                    )
                    for p in range(num_partitions)
                ]
            )

            self.h0 = torch.nn.Buffer(
                torch.zeros(self.n_heads, self.d_head, d_state), persistent=False
            )

            if self.headwise_linear:
                self.proj_zs = nn.ModuleList(
                    [
                        nn.Linear(d_model, self.dim_splits[p], bias=bias)
                        for p in range(num_partitions)
                    ]
                )
                self.proj_xs = nn.ModuleList(
                    [
                        nn.Linear(d_model, self.dim_splits[p], bias=bias)
                        for p in range(num_partitions)
                    ]
                )
                self.proj_dts = nn.ModuleList(
                    [
                        nn.Linear(d_model, self.head_splits[p], bias=bias)
                        for p in range(num_partitions)
                    ]
                )
                self.conv_xs = nn.ModuleList(
                    [
                        _ShortConv(
                            dim=self.dim_splits[p],
                            d_conv=d_conv,
                            bias=conv_bias,
                            activation=activation,
                        )
                        for p in range(num_partitions)
                    ]
                )
                self.dt_biases = nn.ParameterList(
                    [
                        nn.Parameter(torch.rand(self.head_splits[p]))
                        for p in range(num_partitions)
                    ]
                )
                self.A_logs = nn.ParameterList(
                    [
                        nn.Parameter(torch.rand(self.head_splits[p]))
                        for p in range(num_partitions)
                    ]
                )
                self.D_heads = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(self.dim_splits[p]))
                        for p in range(num_partitions)
                    ]
                )
            else:
                self.proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
                self.proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
                self.proj_dt = nn.Linear(d_model, self.n_heads, bias=bias)

                self.conv_x = _ShortConv(
                    dim=self.d_inner,
                    d_conv=d_conv,
                    bias=conv_bias,
                    activation=activation,
                )

                self.dt_bias = nn.Parameter(torch.rand(self.n_heads))
                self.A_log = nn.Parameter(torch.rand(self.n_heads))
                self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = torch.nn.RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.register_load_state_dict_pre_hook(self._load_legacy_state_dict)

    def forward(self, input):
        """
        Input:
            x: [T, D]
        Output:
            y: [T, D]
        """
        if self.head_partitions is None:
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
        else:
            B = self.proj_B(input)
            C = self.proj_C(input)
            B = self.conv_B(B)
            C = self.conv_C(C)

            if not self.headwise_linear:
                z = self.proj_z(input)
                x = self.proj_x(input)
                dt = self.proj_dt(input)

                x = self.conv_x(x)
                dt = F.softplus(dt + self.dt_bias)
                dA = dt * (-torch.exp(self.A_log))

            ys = []
            h0_offset = 0
            dim_offset = 0
            for p, head_partition in enumerate(self.head_partitions):
                n_hp = self.head_splits[p]
                dim_p = self.dim_splits[p]
                h0_p = self.h0[h0_offset : h0_offset + n_hp]

                with vollo_torch.CorePartition(head_partition):
                    if self.headwise_linear:
                        zp = self.proj_zs[p](input)
                        xp = self.proj_xs[p](input)
                        dtp = self.proj_dts[p](input)

                        xp = self.conv_xs[p](xp)

                        dtp = F.softplus(dtp + self.dt_biases[p])
                        dAp = dtp * (-torch.exp(self.A_logs[p]))
                    else:
                        zp = z[:, dim_offset : dim_offset + dim_p]
                        xp = x[:, dim_offset : dim_offset + dim_p]
                        dtp = dt[:, h0_offset : h0_offset + n_hp]
                        dAp = dA[:, h0_offset : h0_offset + n_hp]

                    xp_reshaped = xp.reshape(-1, n_hp, self.d_head)
                    yp = self.ssms[p](
                        [xp_reshaped, B, C, dtp, dAp],
                        h0_p,
                        input_axis=[0] * 5,
                        output_axis=0,
                    )
                    yp = yp.reshape(-1, dim_p)

                    if self.headwise_linear:
                        yp = yp + self.D_heads[p] * xp
                        yp = yp * F.silu(zp)

                    ys.append(yp)

                h0_offset += n_hp
                dim_offset += dim_p

            y = torch.cat(ys, dim=-1)

            if not self.headwise_linear:
                y = y + self.D * x
                y = y * F.silu(z)

            y = self.norm(y)
            return self.out_proj(y)

    def _load_legacy_state_dict(
        self,
        _module,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if self.head_partitions is None:
            # Combine partitioned keys into wide keys if present
            keys_to_combine = [
                ("proj_zs.{}.weight", "proj_z.weight", 0),
                ("proj_zs.{}.bias", "proj_z.bias", 0),
                ("proj_xs.{}.weight", "proj_x.weight", 0),
                ("proj_xs.{}.bias", "proj_x.bias", 0),
                ("proj_dts.{}.weight", "proj_dt.weight", 0),
                ("proj_dts.{}.bias", "proj_dt.bias", 0),
                ("conv_xs.{}.conv1d.conv.weight", "conv_x.conv1d.conv.weight", 0),
                ("conv_xs.{}.conv1d.conv.bias", "conv_x.conv1d.conv.bias", 0),
                ("dt_biases.{}", "dt_bias", 0),
                ("A_logs.{}", "A_log", 0),
                ("D_heads.{}", "D", 0),
            ]
            for head_key_fmt, wide_key, concat_dim in keys_to_combine:
                head_keys = []
                idx = 0
                while prefix + head_key_fmt.format(idx) in state_dict:
                    head_keys.append(prefix + head_key_fmt.format(idx))
                    idx += 1
                if head_keys:
                    values = [state_dict.pop(k) for k in head_keys]
                    state_dict.setdefault(
                        prefix + wide_key, torch.cat(values, dim=concat_dim)
                    )
        else:
            num_partitions = len(self.head_partitions)

            def split_key(wide_key, head_key_fmt, sizes, dim):
                full_wide_key = prefix + wide_key
                if full_wide_key in state_dict:
                    val = state_dict.pop(full_wide_key)
                    chunks = torch.split(val, sizes, dim=dim)
                    for i, chunk in enumerate(chunks):
                        state_dict.setdefault(prefix + head_key_fmt.format(i), chunk)

            if self.headwise_linear:
                split_key("proj_z.weight", "proj_zs.{}.weight", self.dim_splits, dim=0)
                split_key("proj_z.bias", "proj_zs.{}.bias", self.dim_splits, dim=0)
                split_key("proj_x.weight", "proj_xs.{}.weight", self.dim_splits, dim=0)
                split_key("proj_x.bias", "proj_xs.{}.bias", self.dim_splits, dim=0)
                split_key(
                    "proj_dt.weight", "proj_dts.{}.weight", self.head_splits, dim=0
                )
                split_key("proj_dt.bias", "proj_dts.{}.bias", self.head_splits, dim=0)
                split_key(
                    "conv_x.conv1d.conv.weight",
                    "conv_xs.{}.conv1d.conv.weight",
                    self.dim_splits,
                    dim=0,
                )
                split_key(
                    "conv_x.conv1d.conv.bias",
                    "conv_xs.{}.conv1d.conv.bias",
                    self.dim_splits,
                    dim=0,
                )
                split_key("dt_bias", "dt_biases.{}", self.head_splits, dim=0)
                split_key("A_log", "A_logs.{}", self.head_splits, dim=0)
                split_key("D", "D_heads.{}", self.dim_splits, dim=0)


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
        quick_compile=True,
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
        dict(dim=768, state=32, layers=1, fp32=False),
        dict(dim=768, state=32, layers=1, fp32=True),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
