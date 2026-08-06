import warnings
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
        head_partitions: Optional[int] = 6,
        distributed_norm: bool = True,
        no_warning: bool = False,
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
            head_partitions:  Number of head groups to split the heads (and their
                              projections, convolution and scan) into, or None to run
                              all heads as one group. Group `p` is placed on core `p`,
                              so this must not exceed the core count of the target
                              Vollo config. Must be between 1 and the number of
                              heads; warns if it does not divide the heads evenly.
            distributed_norm: Whether to partition the final norm and output
                              projection too, so each core reduces its own slice
                              instead of concatenating it onto one core. Ignored
                              when `head_partitions` is None, where there is only
                              one group and the two forms coincide. Faster, but it
                              splits both reductions into per-core partials that
                              cross cores in bf16, where the unpartitioned form
                              keeps each one in a single wider accumulator.
            no_warning:       Suppress that uneven-split warning, for callers that
                              already know the split is uneven.
        """
        super().__init__()

        self.d_inner = int(expand * d_model)
        self.d_head = d_head
        assert self.d_inner % self.d_head == 0, "d_inner must be divisible by headdim"
        self.n_heads = self.d_inner // self.d_head

        if head_partitions is not None:
            if head_partitions < 1:
                raise ValueError(f"head_partitions must be >= 1, got {head_partitions}")
            if head_partitions > self.n_heads:
                raise ValueError(
                    f"head_partitions ({head_partitions}) must not exceed the number of "
                    f"heads ({self.n_heads}); pass head_partitions=None to run all heads "
                    "as a single group."
                )
            if not no_warning and self.n_heads % head_partitions != 0:
                warnings.warn(
                    f"n_heads ({self.n_heads}) is not a multiple of head_partitions "
                    f"({head_partitions}): the head groups are uneven, so some cores do "
                    "more work than others."
                )

        self.head_partitions = head_partitions
        # With no partitions there is one group spanning every head, so the
        # distributed form is the wide form; keep the wide modules in that case.
        self.distributed_norm = distributed_norm and head_partitions is not None
        self.norm_eps = 1e-5

        self.proj_B = nn.Linear(d_model, d_state, bias=bias)
        self.proj_C = nn.Linear(d_model, d_state, bias=bias)

        self.conv_B = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )
        self.conv_C = _ShortConv(
            dim=d_state, d_conv=d_conv, bias=conv_bias, activation=activation
        )

        # The scan keeps its state flat, as [d_inner! d_state].
        self.h0 = torch.nn.Buffer(torch.zeros(self.d_inner, d_state), persistent=False)

        if self.head_partitions is None:
            self.proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
            self.proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
            self.proj_dt = nn.Linear(d_model, self.n_heads, bias=bias)

            self.conv_x = _ShortConv(
                dim=self.d_inner, d_conv=d_conv, bias=conv_bias, activation=activation
            )

            self.dt_bias = nn.Parameter(torch.rand(self.n_heads))
            self.A_log = nn.Parameter(torch.rand(self.n_heads))

            self.ssm = vollo_torch.nn.Scan(
                _Mamba2Step(
                    n_heads=self.n_heads,
                    d_head=self.d_head,
                    d_state=d_state,
                    fp32=ssm_fp32,
                )
            )

        else:
            # Heads per partition, and the features they span: `h` heads == `d` features.
            self.head_splits = [
                len(chunk)
                for chunk in torch.arange(self.n_heads).tensor_split(
                    self.head_partitions
                )
            ]
            self.dim_splits = [h * self.d_head for h in self.head_splits]

            self.ssms = nn.ModuleList(
                vollo_torch.nn.Scan(
                    _Mamba2Step(
                        n_heads=h, d_head=self.d_head, d_state=d_state, fp32=ssm_fp32
                    )
                )
                for h in self.head_splits
            )
            self.proj_zs = nn.ModuleList(
                nn.Linear(d_model, d, bias=bias) for d in self.dim_splits
            )
            self.proj_xs = nn.ModuleList(
                nn.Linear(d_model, d, bias=bias) for d in self.dim_splits
            )
            self.proj_dts = nn.ModuleList(
                nn.Linear(d_model, h, bias=bias) for h in self.head_splits
            )
            self.conv_xs = nn.ModuleList(
                _ShortConv(dim=d, d_conv=d_conv, bias=conv_bias, activation=activation)
                for d in self.dim_splits
            )
            self.dt_biases = nn.ParameterList(
                nn.Parameter(torch.rand(h)) for h in self.head_splits
            )
            self.A_logs = nn.ParameterList(
                nn.Parameter(torch.rand(h)) for h in self.head_splits
            )

        if self.distributed_norm:
            # Each core holds its slice of the norm gain and the rows of
            # out_proj that its features contract against. The bias is added
            # once, after the partials are summed, so it is not partitioned.
            self.norm_weights = nn.ParameterList(
                nn.Parameter(torch.ones(d)) for d in self.dim_splits
            )
            self.out_projs = nn.ModuleList(
                nn.Linear(d, d_model, bias=False) for d in self.dim_splits
            )
            if bias:
                self.out_bias = nn.Parameter(torch.zeros(d_model))
            else:
                self.out_bias = None
        else:
            self.norm = torch.nn.RMSNorm(self.d_inner, eps=self.norm_eps)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.register_load_state_dict_pre_hook(self._load_legacy_state_dict)

    def forward(self, input):
        """
        Input:
            x: [T, D]
        Output:
            y: [T, D]
        """
        B = self.proj_B(input)
        C = self.proj_C(input)
        B = self.conv_B(B)
        C = self.conv_C(C)

        if self.head_partitions is None:
            x = self.conv_x(self.proj_x(input))

            dt = F.softplus(self.proj_dt(input) + self.dt_bias)
            dA = dt * (-torch.exp(self.A_log))

            # The step folds the D skip connection into its output.
            y = self.ssm([x, B, C, dt, dA], self.h0, input_axis=[0] * 5, output_axis=0)

            # Gating (FLA always uses SiLU regardless of hidden_act)
            # This is "norm after gate" configuration.
            y = y * torch.nn.functional.silu(self.proj_z(input))

            y = self.norm(y)

            return self.out_proj(y)
        else:
            # RMSNorm's reduction spans every partition, so it cannot simply be
            # done per core. But its scale is a per-timestep scalar and out_proj
            # is linear, so under `distributed_norm` the scale factors out of the
            # projection:
            #
            #   out = W @ (y * w / rms) = (1 / rms) * sum_p (y_p * w_p) @ W_p
            #
            # Each core then contributes its slice's sum of squares and a partial
            # projection, and only those cross cores.
            ys = []
            ss_parts = []
            partials = []
            dim_offset = 0
            for p in range(self.head_partitions):
                dim_p = self.dim_splits[p]
                h0_p = self.h0[dim_offset : dim_offset + dim_p]

                # One head group per core.
                with vollo_torch.CorePartition([p]):
                    xp = self.conv_xs[p](self.proj_xs[p](input))

                    dtp = F.softplus(self.proj_dts[p](input) + self.dt_biases[p])
                    dAp = dtp * (-torch.exp(self.A_logs[p]))

                    yp = self.ssms[p](
                        [xp, B, C, dtp, dAp],
                        h0_p,
                        input_axis=[0] * 5,
                        output_axis=0,
                    )

                    yp = yp * F.silu(self.proj_zs[p](input))

                    if self.distributed_norm:
                        ss_parts.append((yp * yp).sum(-1, keepdim=True))
                        partials.append(self.out_projs[p](yp * self.norm_weights[p]))
                    else:
                        ys.append(yp)

                dim_offset += dim_p

            if not self.distributed_norm:
                y = torch.cat(ys, dim=-1)
                return self.out_proj(self.norm(y))

            out = _tree_sum(partials)

            # Each core's sum of squares is a single element, so gathering them
            # into one vector and reducing that costs one cross-core
            # concatenation plus one reduction, rather than a chain of cross-core
            # adds.
            ss = torch.cat(ss_parts, dim=-1).sum(-1, keepdim=True)

            out = out * torch.rsqrt(ss / self.d_inner + self.norm_eps)

            if self.out_bias is not None:
                out = out + self.out_bias

            return out

    # Format:
    #   checkpoint key,
    #   unpartitioned key,
    #   per-partition key,
    #   whether the tensor is split per feature or per head,
    #   the axis it splits along

    _PARTITIONED_KEYS = (
        ("proj_z.weight", "proj_z.weight", "proj_zs.{}.weight", "dim", 0),
        ("proj_z.bias", "proj_z.bias", "proj_zs.{}.bias", "dim", 0),
        ("proj_x.weight", "proj_x.weight", "proj_xs.{}.weight", "dim", 0),
        ("proj_x.bias", "proj_x.bias", "proj_xs.{}.bias", "dim", 0),
        ("proj_dt.weight", "proj_dt.weight", "proj_dts.{}.weight", "head", 0),
        ("proj_dt.bias", "proj_dt.bias", "proj_dts.{}.bias", "head", 0),
        (
            "conv_x.conv1d.conv.weight",
            "conv_x.conv1d.conv.weight",
            "conv_xs.{}.conv1d.conv.weight",
            "dim",
            0,
        ),
        (
            "conv_x.conv1d.conv.bias",
            "conv_x.conv1d.conv.bias",
            "conv_xs.{}.conv1d.conv.bias",
            "dim",
            0,
        ),
        ("dt_bias", "dt_bias", "dt_biases.{}", "head", 0),
        ("A_log", "A_log", "A_logs.{}", "head", 0),
        ("D", "ssm.step.D", "ssms.{}.step.D", "dim", 0),
    )

    _NORM_KEYS = (
        ("norm.weight", "norm.weight", "norm_weights.{}", "dim", 0),
        ("out_proj.weight", "out_proj.weight", "out_projs.{}.weight", "dim", 1),
    )

    # Keys that change name between the layouts but are not split
    _RENAMED_KEYS = (("out_proj.bias", "out_bias"),)

    def _load_legacy_state_dict(self, _module, state_dict, prefix, *_hook_args):
        """
        Translate a checkpoint between the wide and per-partition key layouts, so one
        saved from any configuration -- with any number of partitions -- loads into
        any other.
        """
        self._split_or_merge(
            state_dict, prefix, self._PARTITIONED_KEYS, self.head_partitions is not None
        )
        self._split_or_merge(state_dict, prefix, self._NORM_KEYS, self.distributed_norm)

        for wide, part in self._RENAMED_KEYS:
            src, dst = (wide, part) if self.distributed_norm else (part, wide)
            if prefix + src in state_dict:
                state_dict.setdefault(prefix + dst, state_dict.pop(prefix + src))

    def _split_or_merge(self, state_dict, prefix, keys, to_partitioned: bool):
        for ckpt, wide, part_fmt, split_by, axis in keys:
            # Gather whichever layout the checkpoint used into one wide tensor.
            tensor = None
            for key in dict.fromkeys((prefix + ckpt, prefix + wide)):
                if key in state_dict:
                    tensor = state_dict.pop(key)
                    break
            else:
                parts, idx = [], 0
                while prefix + part_fmt.format(idx) in state_dict:
                    parts.append(state_dict.pop(prefix + part_fmt.format(idx)))
                    idx += 1
                if parts:
                    tensor = torch.cat(parts, dim=axis)

            if tensor is None:
                continue

            # ... then split it the way this model holds it.
            if not to_partitioned:
                state_dict.setdefault(prefix + wide, tensor)
            else:
                sizes = self.head_splits if split_by == "head" else self.dim_splits
                for i, chunk in enumerate(torch.split(tensor, sizes, dim=axis)):
                    state_dict.setdefault(prefix + part_fmt.format(i), chunk)


def _tree_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """
    Sum per-core partials pairwise rather than in a chain, so the cross-core
    reduction is log-depth: the partials become available in parallel, and a
    chain would serialise them behind each other.
    """
    while len(parts) > 1:
        parts = [
            parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
            for i in range(0, len(parts), 2)
        ]
    return parts[0]


class _Mamba2Step(nn.Module):
    @beartype
    def __init__(self, n_heads: int, d_head: int, d_state: int, fp32: bool):
        super().__init__()

        self.n_heads = n_heads  # h
        self.d_head = d_head  # p
        self.d_state = d_state  # n
        self.d_inner = n_heads * d_head  # d

        self.fp_context = vollo_torch.Fp32Activations if fp32 else nullcontext

        # The skip connection's scale. It belongs to the step, rather than to the
        # enclosing Mamba2, because the step folds it into the instantaneous term
        # below, and a parameter read inside a Scan must be owned by the step
        # module. `_PARTITIONED_KEYS` maps the usual `D` checkpoint key onto it.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # head_expand[i, g] = 1 iff feature i belongs to head g, so that a
        # per-head scalar can be broadcast over its features by contracting the
        # data dimension instead of moving it -- see _broadcast_heads.
        expand = torch.zeros(self.d_inner, n_heads)
        for i in range(self.d_inner):
            expand[i, i // d_head] = 1.0
        self.head_expand = torch.nn.Buffer(expand, persistent=False)

    def forward(self, inputs: list[torch.Tensor], S: torch.Tensor):
        """
        The state is held flat, as `[d! n]`: Vollo's matrix-vector product wants
        the matrix's data dimension second-innermost, so this is the layout the
        state read compiles to natively.

        Inputs:
                 x: [d!]
              B, C: [n!]
            dA, dt: [h!]
                 S: [d! n]

        Returns:
                 y: [d!]
        """
        x, B, C, dt, dA = inputs

        with self.fp_context():
            dA = dA.exp()

        # dA is the output of an fp32 op, so it is broadcast by slicing the data
        # dimension apart (matmul would decay to bf16).
        dA = self._broadcast_heads(dA)  # [h!] -> [d!]

        # [d h] @ [h! 1] -> [d! 1] -> [d!]
        dt = (self.head_expand @ dt.unsqueeze(-1)).squeeze(-1)

        # The instantaneous term folds in the D skip connection, since
        #     dt * x * (B . C) + D * x == x * (dt * (B . C) + D)
        # which leaves just one multiply and one add waiting on x.
        instant = x * (dt * (B * C).sum(-1, keepdim=True) + self.D)

        # [d! 1] * [1! n] -> [d! n]
        dB = (dt * x).unsqueeze(-1) * torch.stack(
            [B[i : i + 1] for i in range(self.d_state)], dim=1
        )

        # [d! n] @ [n!] -> [d!]
        y = dA * (S @ C) + instant

        with self.fp_context():
            S = dA.unsqueeze(-1) * S + dB

        return y, S

    def _broadcast_heads(self, v: torch.Tensor) -> torch.Tensor:
        """
        [h!] -> [d!]: broadcast each per-head scalar over that head's features.
        """
        return torch.cat(
            [
                torch.broadcast_to(v[g : g + 1], (self.d_head,))
                for g in range(self.n_heads)
            ],
            dim=-1,
        )


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
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    # Head group `p` runs on core `p`, so use one group per core of this config.
    partitions = CONFIGS[config].num_cores

    input = torch.randn(2, dim)

    model = nn.Sequential().extend(
        Mamba2(
            d_model=dim,
            d_state=state,
            ssm_fp32=fp32,
            head_partitions=partitions,
            no_warning=True,
        )
        for _ in range(layers)
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
        # Note: some of these configurations are sub-optimal for core splitting
        # because the number of heads is not divisible by the number of cores,
        # but they are left unmodified so that we can compare with historical
        # data
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
