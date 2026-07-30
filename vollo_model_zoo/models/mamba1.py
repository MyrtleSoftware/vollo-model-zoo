import math
from pathlib import Path
from typing import Literal, Optional

import torch
import vollo_torch
from beartype import beartype
from beartype.typing import Generator
from torch import nn


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
            d_state:          SSM state multiplier factor (i.e. state is [d_state, d_expand * d_model]).
            d_conv:           Local convolution width.
            expand:           Hidden state expansion factor.
            dt_rank:          Generalized delta dimension.
            bias:             Input/output projection bias.
            conv_bias:        Convolutional bias.
            activation:       Activation function to use for convolution/gate.
            head_partitions:  Sequence of core partition indices, one per head.
            headwise_linear:  Whether to run linear projections per head inside core partitions.
        """

        super().__init__()

        self.bias = bias
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        if head_partitions is not None:
            num_heads = len(head_partitions)
            if self.d_inner % num_heads != 0:
                if head_partitions == ((0,), (1,), (2,), (3,), (4,), (5,)):
                    head_partitions = None
                else:
                    raise ValueError(
                        f"Expected d_inner ({self.d_inner}) to be divisible by len(head_partitions) ({num_heads})"
                    )

        self.head_partitions = head_partitions
        self.headwise_linear = self.head_partitions is not None and headwise_linear

        match activation:
            case "silu":
                self.act = nn.SiLU()
            case "relu":
                self.act = nn.ReLU()

        if self.head_partitions is None:
            step = _MambaStep(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dt_rank=dt_rank,
            )

            self.ssm = vollo_torch.nn.Scan(step)
            self.h0 = torch.nn.Buffer(step._init_state(), persistent=False)

            self.in_proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
            self.in_proj_z = nn.Linear(d_model, self.d_inner, bias=bias)

            self.conv1d = vollo_torch.nn.PaddedConv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                kernel_size=d_conv,
                bias=conv_bias,
            )
        else:
            num_heads = len(self.head_partitions)
            head_dim = self.d_inner // num_heads

            steps = [
                _MambaStep(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    dt_rank=dt_rank,
                    d_inner=head_dim,
                )
                for _ in range(num_heads)
            ]

            self.ssms = nn.ModuleList([vollo_torch.nn.Scan(s) for s in steps])
            self.h0 = torch.nn.Buffer(
                torch.stack([s._init_state() for s in steps]), persistent=False
            )

            if self.headwise_linear:
                self.in_proj_xs = nn.ModuleList(
                    [nn.Linear(d_model, head_dim, bias=bias) for _ in range(num_heads)]
                )
                self.in_proj_zs = nn.ModuleList(
                    [nn.Linear(d_model, head_dim, bias=bias) for _ in range(num_heads)]
                )
                self.conv1ds = nn.ModuleList(
                    [
                        vollo_torch.nn.PaddedConv1d(
                            in_channels=head_dim,
                            out_channels=head_dim,
                            groups=head_dim,
                            kernel_size=d_conv,
                            bias=conv_bias,
                        )
                        for _ in range(num_heads)
                    ]
                )
            else:
                self.in_proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
                self.in_proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
                self.conv1d = vollo_torch.nn.PaddedConv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    kernel_size=d_conv,
                    bias=conv_bias,
                )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.register_load_state_dict_pre_hook(self._load_legacy_state_dict)

    def forward(self, x):
        """
        x: [time, d_model]

        Returns r: [time, d_model]
        """
        if self.head_partitions is None:
            # Up projection
            x_proj, z_proj = self.in_proj_x(x), self.in_proj_z(x)

            # Vollo requires that time is rightmost dimension for convolution
            x_proj = x_proj.transpose(0, 1)
            x_proj = self.act(self.conv1d(x_proj))
            x_proj = x_proj.transpose(0, 1)

            y = self.ssm(x_proj, self.h0, input_axis=0, output_axis=0)  # [t, D]

            # Outer residual/gate
            y = y * self.act(z_proj)

            # Down projection
            y = self.out_proj(y)  # [t, d]
            return y
        else:
            if not self.headwise_linear:
                x_proj, z_proj = self.in_proj_x(x), self.in_proj_z(x)
                x_proj = x_proj.transpose(0, 1)
                x_proj = self.act(self.conv1d(x_proj))
                x_proj = x_proj.transpose(0, 1)

            num_heads = len(self.head_partitions)
            head_dim = self.d_inner // num_heads

            ys = []
            for head_ix, head_partition in enumerate(self.head_partitions):
                with vollo_torch.CorePartition(head_partition):
                    if self.headwise_linear:
                        xh = self.in_proj_xs[head_ix](x)
                        zh = self.in_proj_zs[head_ix](x)

                        xh = xh.transpose(0, 1)
                        xh = self.act(self.conv1ds[head_ix](xh))
                        xh = xh.transpose(0, 1)
                    else:
                        xh = x_proj[:, head_ix * head_dim : (head_ix + 1) * head_dim]
                        zh = z_proj[:, head_ix * head_dim : (head_ix + 1) * head_dim]

                    yh = self.ssms[head_ix](
                        xh, self.h0[head_ix], input_axis=0, output_axis=0
                    )
                    yh = yh * self.act(zh)
                    ys.append(yh)

            y = torch.cat(ys, dim=-1)
            y = self.out_proj(y)
            return y

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
            # Combine partitioned keys into unpartitioned keys if present
            # ssms.{i}.step.* -> ssm.step.*
            keys_to_combine = [
                ("ssms.{}.step.A_log_t", "ssm.step.A_log_t", 1),
                ("ssms.{}.step.x_proj_t.weight", "ssm.step.x_proj_t.weight", 1),
                ("ssms.{}.step.x_proj_B.weight", "ssm.step.x_proj_B.weight", 1),
                ("ssms.{}.step.x_proj_C.weight", "ssm.step.x_proj_C.weight", 1),
                ("ssms.{}.step.dt_proj.weight", "ssm.step.dt_proj.weight", 0),
                ("ssms.{}.step.dt_proj.bias", "ssm.step.dt_proj.bias", 0),
                ("ssms.{}.step.D", "ssm.step.D", 1),
                ("in_proj_xs.{}.weight", "in_proj_x.weight", 0),
                ("in_proj_xs.{}.bias", "in_proj_x.bias", 0),
                ("in_proj_zs.{}.weight", "in_proj_z.weight", 0),
                ("in_proj_zs.{}.bias", "in_proj_z.bias", 0),
                ("conv1ds.{}.conv.weight", "conv1d.conv.weight", 0),
                ("conv1ds.{}.conv.bias", "conv1d.conv.bias", 0),
            ]
            for head_key_fmt, wide_key, concat_dim in keys_to_combine:
                # Find how many head keys exist
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
            num_heads = len(self.head_partitions)
            head_dim = self.d_inner // num_heads

            def split_key(wide_key, head_key_fmt, sizes, dim):
                full_wide_key = prefix + wide_key
                if full_wide_key in state_dict:
                    val = state_dict.pop(full_wide_key)
                    chunks = torch.split(val, sizes, dim=dim)
                    for i, chunk in enumerate(chunks):
                        state_dict.setdefault(prefix + head_key_fmt.format(i), chunk)

            split_key(
                "ssm.step.A_log_t",
                "ssms.{}.step.A_log_t",
                [head_dim] * num_heads,
                dim=1,
            )
            split_key(
                "ssm.step.x_proj_t.weight",
                "ssms.{}.step.x_proj_t.weight",
                [head_dim] * num_heads,
                dim=1,
            )
            split_key(
                "ssm.step.x_proj_B.weight",
                "ssms.{}.step.x_proj_B.weight",
                [head_dim] * num_heads,
                dim=1,
            )
            split_key(
                "ssm.step.x_proj_C.weight",
                "ssms.{}.step.x_proj_C.weight",
                [head_dim] * num_heads,
                dim=1,
            )
            split_key(
                "ssm.step.dt_proj.weight",
                "ssms.{}.step.dt_proj.weight",
                [head_dim] * num_heads,
                dim=0,
            )
            split_key(
                "ssm.step.dt_proj.bias",
                "ssms.{}.step.dt_proj.bias",
                [head_dim] * num_heads,
                dim=0,
            )
            split_key("ssm.step.D", "ssms.{}.step.D", [head_dim] * num_heads, dim=1)

            if self.headwise_linear:
                split_key(
                    "in_proj_x.weight",
                    "in_proj_xs.{}.weight",
                    [head_dim] * num_heads,
                    dim=0,
                )
                split_key(
                    "in_proj_x.bias",
                    "in_proj_xs.{}.bias",
                    [head_dim] * num_heads,
                    dim=0,
                )
                split_key(
                    "in_proj_z.weight",
                    "in_proj_zs.{}.weight",
                    [head_dim] * num_heads,
                    dim=0,
                )
                split_key(
                    "in_proj_z.bias",
                    "in_proj_zs.{}.bias",
                    [head_dim] * num_heads,
                    dim=0,
                )
                split_key(
                    "conv1d.conv.weight",
                    "conv1ds.{}.conv.weight",
                    [head_dim] * num_heads,
                    dim=0,
                )
                split_key(
                    "conv1d.conv.bias",
                    "conv1ds.{}.conv.bias",
                    [head_dim] * num_heads,
                    dim=0,
                )


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
        d_inner: int | None = None,
    ):
        """
        Second half of mamba (post convolution) mainly the SSM.
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = (
            d_inner if d_inner is not None else int(self.expand * self.d_model)
        )
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
            self.d_inner,
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


@beartype
def _vm(
    dim: int,
    state: int,
    layers: int,
    config: str,
    head_partitions: Optional[tuple[tuple[int, ...], ...]] = (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
    ),
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, dim)

    model = torch.nn.Sequential().extend(
        Mamba(
            d_model=dim,
            d_state=state,
            head_partitions=head_partitions,
        )
        for _ in range(layers)
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        quick_compile=True,
        meta=dict(
            dim=dim,
            state=state,
            layers=layers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, state=6, layers=1),
        dict(dim=32 * 12, state=6 * 2, layers=1),
        dict(dim=32 * 12, state=6 * 2, layers=2),
        dict(
            dim=32 * 32,
            state=6 * 4,
            layers=1,
            head_partitions=((0, 1), (2, 3), (4,), (5,)),
        ),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
