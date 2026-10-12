from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
from beartype import beartype
from torch import nn
from vollo_torch import CorePartition
from vollo_torch.nn import PaddedConv1d


def _pad(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Zero border of one on each of `dims` (all non-data axes)."""
    for dim in dims:
        z = x.narrow(dim, 0, 1) * 0
        x = torch.cat([z, x, z], dim=dim)
    return x


class _Conv2d(nn.Module):
    """
    Exact `nn.Conv2d(in, out, 3, padding=1)` + ReLU from ops Vollo supports.

    Activations are a batch of grid rows [T, C!, S]: rounds on the batch axis,
    stabilisers along the sequence axis, channels the data dimension. Kernel row dt
    is a non-streaming `Conv1d` (kernel 3 along the stabilisers, over a zero column
    either side) and the three are summed: the bf16 sum runs in a different order
    to `Conv2d`'s. The compiler skips the multiplications by the zero columns, but
    not by a whole zero row, so the top and bottom kernel rows only run over the
    T - 1 output rows they reach, padded with a zero row before the sum. `Conv1d`
    rows are `weight[:, :, dt]` of the `Conv2d`; the bias sits on the centre row,
    the only one that reaches every output row.

    With `cores`, kernel row dt runs on `cores[dt]` (a `CorePartition`), which
    splits the weights three ways rather than copying them to every core.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rounds: int,
        cores: list[list[int]] | None = None,
    ):
        super().__init__()
        assert rounds >= 2
        self.rounds, self.cores = rounds, cores
        self.rows = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, 3, bias=(dt == 1)) for dt in range(3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[T, in!, S] -> [T, out!, S]"""
        T = self.rounds
        x = _pad(x, (2,))  # [T, in!, S+2]
        # Kernel row dt reads input row t + dt - 1 for output row t.
        inputs = [x[: T - 1], x, x[1:]]  # output rows 1.., all, ..T-2
        ys = []
        for dt, row in enumerate(self.rows):
            with CorePartition(self.cores[dt]) if self.cores else nullcontext():
                ys.append(row(inputs[dt]))  # [T or T-1, out!, S]
        z = ys[0][:1] * 0  # [1, out!, S]
        top = torch.cat([z, ys[0]], dim=0)
        bottom = torch.cat([ys[2], z], dim=0)
        return torch.relu(top + ys[1] + bottom)


class FiLMDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        num_stabilisers: int,
        rounds: int,
        channels: list[int] = [128, 256, 512],
        num_cores: int | None = None,
    ):
        """
        Calibration-conditioned QEC decoder over the whole syndrome window.

        Stein et al. 2026, https://arxiv.org/abs/2601.16123, as published. A
        repetition code of distance d has d-1 stabilisers measured over r rounds;
        the decision is made once, after the last round (`time_axis=None`). A stack
        of 3x3 convolutions (padding 1) over the (rounds x stabilisers) grid of
        detection events, each block computing `relu(gamma * conv(x) + beta)`, is
        flattened into a dense head giving a flip logit per data qubit (d). gamma
        and beta come from the device's calibration data (T1, T2, gate and readout
        errors) via a graph network that runs offline, once per calibration; here
        `fold_film` bakes them into the conv weights, so they cost nothing.

        Vollo has no `Conv2d` and cannot reshape the data dimension, so both are
        rewritten exactly (see `_Conv2d`): the first layer's single input channel
        is unfolded into its 9 neighbours and given one `Linear`, and the flatten
        plus dense head is one `Conv1d` whose kernel spans all r * (d-1) positions.
        `load_reference` loads the weights of the plain `Conv2d` / `Linear` form.

        Args:
            num_stabilisers: Syndrome bits per round (d - 1)
            rounds:          Rounds r in the window
            channels:        Output channels of each block (the paper's 128, 256, 512)
            num_cores:       Cores of the target config. A multiple of 3 spreads
                             the three kernel rows of every conv block over a third
                             of the cores each; otherwise placement is left to the
                             compiler.
        """
        super().__init__()
        self.rounds, self.stabs = rounds, num_stabilisers
        cores = None
        if num_cores is not None and num_cores % 3 == 0:
            k = num_cores // 3
            cores = [list(range(dt * k, (dt + 1) * k)) for dt in range(3)]
        self.first = nn.Linear(9, channels[0])
        self.convs = nn.ModuleList(
            _Conv2d(i, o, rounds, cores) for i, o in zip(channels[:-1], channels[1:])
        )
        self.head = nn.Conv1d(
            channels[-1], num_stabilisers + 1, rounds * num_stabilisers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (1, rounds, num_stabilisers, 1): detection events

        Returns:
            Tensor of shape (1, 1, num_stabilisers + 1): data-qubit flip logits
        """
        T, S = self.rounds, self.stabs
        xp = _pad(x, (2, 1))  # [1, T+2, S+2, 1!]
        taps = [xp[:, i : i + T, j : j + S] for i in range(3) for j in range(3)]
        x = torch.relu(self.first(torch.cat(taps, dim=3)))  # [1, T, S, C!]
        x = x[0].transpose(1, 2)  # [T, C!, S]
        for conv in self.convs:
            x = conv(x)  # [T, C!, S]
        x = x.transpose(1, 2).reshape(1, T * S, -1)  # [1, T*S, C!]
        return self.head(x.transpose(1, 2)).transpose(1, 2)  # [1, 1, d!]

    @torch.no_grad()
    def load_reference(self, convs: list[nn.Conv2d], head: nn.Linear):
        """Load the plain form: `Conv2d(., ., 3, padding=1)` blocks, then
        `Linear` over the flattened (channels, rounds, stabilisers) output."""
        self.first.weight.copy_(convs[0].weight.flatten(1))
        self.first.bias.copy_(convs[0].bias)
        for conv, ref in zip(self.convs, convs[1:]):
            for dt, row in enumerate(conv.rows):
                row.weight.copy_(ref.weight[:, :, dt])
            conv.rows[1].bias.copy_(ref.bias)
        self.head.weight.copy_(head.weight.view(self.head.weight.shape))
        self.head.bias.copy_(head.bias)

    @torch.no_grad()
    def fold_film(self, gamma: list[torch.Tensor], beta: list[torch.Tensor]):
        """Bake per-block gamma, beta (each of shape (channels[i],)) into the convs."""
        first_and_rows = [self.first] + [c.rows[1] for c in self.convs]
        for layer, g, b in zip(first_and_rows, gamma, beta):
            layer.bias.mul_(g).add_(b)
        self.first.weight.mul_(gamma[0][:, None])
        for conv, g in zip(self.convs, gamma[1:]):
            for row in conv.rows:
                row.weight.mul_(g[:, None, None])


class _CoreSplitConv(nn.Module):
    """
    A `PaddedConv1d` + ReLU whose output channels are split into groups, group g
    on core g (a `CorePartition`), so an optional FiLM modulation after it runs on
    the core that produced each slice.
    """

    def __init__(
        self, in_channels: int, sizes: list[int], kernel_size: int, dilation: int
    ):
        super().__init__()
        self.sizes = sizes
        self.parts = nn.ModuleList(
            PaddedConv1d(in_channels, z, kernel_size, dilation=dilation) for z in sizes
        )

    def forward(self, x, gamma=None, beta=None):
        """[B, in!, T] (and gamma, beta [B, out!, T]) -> [B, out!, T]"""
        ys, a = [], 0
        for g, (part, z) in enumerate(zip(self.parts, self.sizes)):
            with CorePartition([g]):
                y = part(x)  # [B, z!, T]
                if gamma is not None:
                    y = gamma[:, a : a + z] * y + beta[:, a : a + z]
                ys.append(torch.relu(y))
            a += z
        return torch.cat(ys, dim=1)

    @torch.no_grad()
    def load_conv(self, conv: PaddedConv1d):
        """Load the weights of the unsplit `PaddedConv1d`."""
        a = 0
        for part, z in zip(self.parts, self.sizes):
            part.conv.weight.copy_(conv.conv.weight[a : a + z])
            part.conv.bias.copy_(conv.conv.bias[a : a + z])
            a += z


class StreamingFiLMDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        num_stabilisers: int,
        channels: list[int],
        kernel_size: int,
        dilation: int,
        film_input: bool = False,
        core_groups: list[int] | None = None,
        first_core: int | None = None,
    ):
        """
        Causal, round-streaming redesign of `FiLMDecoder` (needs retraining).

        Not the paper's network but a Vollo-oriented variant: a causal 1-D
        convolution over rounds, with the d-1 stabilisers folded into channels, so
        each inference consumes one round and outputs a flip logit per data qubit
        (d). Each block computes `relu(gamma * conv(x) + beta)` as in the paper.

        Two ways to apply that FiLM modulation:

        - `film_input=False` (deployed): `fold_film` bakes gamma and beta into the
          conv weights and biases at calibration time, so it costs nothing per round.
        - `film_input=True`: gamma and beta are extra input channels, sent with every
          round's syndrome, laid out as [gamma_1, beta_1, gamma_2, beta_2, ...].

        Args:
            num_stabilisers: Syndrome bits per round (d - 1)
            channels:        Output channels of each block
            kernel_size:     Temporal kernel size
            dilation:        Temporal dilation. Two blocks reach 1 + 2 * (kernel_size - 1)
                             * dilation rounds back, which must cover the r rounds of the
                             experiment (r <= 11 in the paper). Dilation is free on Vollo,
                             while a third block adds ~0.15 us of dependency depth.
            film_input:      Take gamma and beta as part of the input
            core_groups:     Split the last block's output channels into groups of
                             these sizes, group g on core g (see `_CoreSplitConv`).
                             With `film_input`, each core then applies the FiLM
                             modulation to its own slice rather than one core doing
                             it all. `load_conv` loads the unsplit weights.
            first_core:      Pin the first block to this core. Off core 0, where
                             the compiler starts placing every block, the small
                             first block stops queueing behind the larger second
                             one (any other core measured the same, folded).
        """
        super().__init__()
        self.num_stabilisers = num_stabilisers
        self.channels = channels
        self.film_input = film_input
        self.first_core = first_core
        ins = [num_stabilisers] + channels[:-1]
        self.convs = nn.ModuleList(
            PaddedConv1d(i, o, kernel_size, dilation=dilation)
            for i, o in zip(ins, channels)
        )
        if core_groups is not None:
            assert sum(core_groups) == channels[-1]
            self.convs[-1] = _CoreSplitConv(ins[-1], core_groups, kernel_size, dilation)
        self.head = nn.Linear(channels[-1], num_stabilisers + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, num_stabilisers [+ 2 * sum(channels)], Time)

        Returns:
            Tensor of shape (Batch, Time, num_stabilisers + 1): data-qubit flip logits
        """
        if self.film_input:
            film = x[:, self.num_stabilisers :]  # [B, 2*sum(channels)!, T]
            x = x[:, : self.num_stabilisers]  # [B, S!, T]
        offset = 0
        for i, (conv, c) in enumerate(zip(self.convs, self.channels)):
            gamma = beta = None
            if self.film_input:
                gamma = film[:, offset : offset + c]  # [B, c!, T]
                beta = film[:, offset + c : offset + 2 * c]
                offset += 2 * c
            if isinstance(conv, _CoreSplitConv):
                x = conv(x, gamma, beta)  # [B, c!, T]
                continue
            pin = i == 0 and self.first_core is not None
            with CorePartition([self.first_core]) if pin else nullcontext():
                x = conv(x)  # [B, c!, T]
                if gamma is not None:
                    x = gamma * x + beta
                x = torch.relu(x)
        return self.head(x.transpose(1, 2))  # [B, T, d!]

    @torch.no_grad()
    def fold_film(self, gamma: list[torch.Tensor], beta: list[torch.Tensor]):
        """Bake per-block gamma, beta (each of shape (channels[i],)) into the convs."""
        for conv, g, b in zip(self.convs, gamma, beta):
            a = 0
            for part in conv.parts if isinstance(conv, _CoreSplitConv) else [conv]:
                z = part.conv.out_channels
                part.conv.weight.mul_(g[a : a + z, None, None])
                part.conv.bias.mul_(g[a : a + z]).add_(b[a : a + z])
                a += z


@beartype
def _core_groups(channels: int, num_cores: int, block_size: int) -> list[int]:
    """`channels` split into near-equal whole-block groups, at most one per core."""
    blocks = -(-channels // block_size)
    sizes = [blocks // num_cores + (i < blocks % num_cores) for i in range(num_cores)]
    groups = [z * block_size for z in sizes if z]
    groups[-1] -= sum(groups) - channels
    return groups


@beartype
def _spread_cores(
    config: str, num_stabilisers: int, rounds: int, channels: list[int]
) -> int | None:
    """
    `num_cores` for `FiLMDecoder` on `config`, or None to leave placement to the
    compiler.

    By default the compiler copies the conv weights to every core and splits the
    grid positions between them. Where a copy per core plus the head does not fit
    the cores' weight stores, spreading the kernel rows over core thirds splits
    the conv weights three ways instead.
    """
    from vollo_model_zoo.vm import CONFIGS

    hw = CONFIGS[config]
    blocks = lambda i, o: -(-i // hw.block_size) * -(-o // hw.block_size)
    conv = sum(9 * blocks(i, o) for i, o in zip(channels[:-1], channels[1:]))
    head = rounds * num_stabilisers * blocks(channels[-1], num_stabilisers + 1)
    if hw.num_cores * conv + head <= hw.num_cores * hw.weight_store_depth:
        return None
    return hw.num_cores


@beartype
def _vm(
    num_stabilisers: int,
    channels: list[int],
    config: str,
    rounds: int | None = None,
    kernel_size: int = 3,
    dilation: int = 3,
    film_input: bool = False,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    meta = dict(distance=num_stabilisers + 1, channels=str(channels))
    if rounds is not None:
        # The paper's decoder: one decision per window of `rounds` rounds.
        input = torch.randn(1, rounds, num_stabilisers, 1)
        model = FiLMDecoder(
            num_stabilisers,
            rounds,
            channels,
            _spread_cores(config, num_stabilisers, rounds, channels),
        )
        time_axis = None
        meta |= dict(model="window", rounds=rounds)
    else:
        # With film_input, gamma and beta ride along as extra input channels.
        num_inputs = num_stabilisers + 2 * sum(channels) * film_input
        input = torch.randn(1, num_inputs, 8)
        # The first block runs on core 1; with film_input the last block is also
        # split over every core, so each core modulates its own slice.
        hw = CONFIGS[config]
        core_groups = None
        if film_input:
            core_groups = _core_groups(channels[-1], hw.num_cores, hw.block_size)
        first_core = 1
        model = StreamingFiLMDecoder(
            num_stabilisers,
            channels,
            kernel_size,
            dilation,
            film_input,
            core_groups,
            first_core,
        )
        time_axis = 2
        meta |= dict(
            model="streaming",
            kernel=kernel_size,
            dilation=dilation,
            film="input" if film_input else "folded",
        )

    # quick_compile: the optimiser passes cost minutes on the window models
    # and leave the latency unchanged.
    return vollo_info(
        model,
        input,
        config=config,
        time_axis=time_axis,
        quick_compile=True,
        meta=meta,
    )


@beartype
def main(config: str = "V80") -> Generator:
    # Streaming redesign: deployment sizes (d = 5, 7, 11) are sized to fit a ~1 us
    # syndrome round. Baseline: 10 * 384 * 5 + 384 * 512 * 5 = 1.00M conv weights,
    # plus a 512 * 11 head, ~1.01M. Its contiguous latency sits right at the round
    # time. The paper's whole-window decoder comes last, as its reaction latency
    # after the final round is tens of us, and (d=11) its weights are 2.1M.
    for x in [
        dict(num_stabilisers=4, channels=[128, 256]),
        dict(num_stabilisers=6, channels=[256, 512]),
        dict(num_stabilisers=10, channels=[256, 512]),
        dict(num_stabilisers=10, channels=[384, 512], kernel_size=5, dilation=2),
        dict(num_stabilisers=10, channels=[256, 512], film_input=True),
        dict(num_stabilisers=4, channels=[128, 256, 512], rounds=5),
        dict(num_stabilisers=10, channels=[128, 256, 512], rounds=11),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
