from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


def _pad(x: torch.Tensor) -> torch.Tensor:
    """Zero border of one on both spatial axes: [1, T, S, C!] -> [1, T+2, S+2, C!]."""
    z = x[:, :, :1] * 0
    x = torch.cat([z, x, z], dim=2)
    z = x[:, :1] * 0
    return torch.cat([z, x, z], dim=1)


class _Conv2d(nn.Module):
    """
    Exact `nn.Conv2d(in, out, 3, padding=1)` (+ ReLU) from ops Vollo supports.

    Activations live as a flattened, zero-padded grid [1, in!, (T+2)*(S+2)], row
    after row. A shift by one row is then a slice offset of S+2, so each of the 3
    kernel rows is one non-streaming `Conv1d` (kernel 3, data dimension on channels)
    over a row-shifted slice, and the three are summed. The two padding columns of
    each row come out as garbage and are zeroed by a constant mask, which also
    restores the zero border the next layer needs. `Conv1d` rows are `weight[:, :,
    dt]` of the `Conv2d`; the bias sits on the first row only.
    """

    def __init__(self, in_channels: int, out_channels: int, rounds: int, stabs: int):
        super().__init__()
        self.rounds, self.width = rounds, stabs + 2
        self.rows = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, 3, bias=(dt == 0)) for dt in range(3)
        )
        n = rounds * self.width
        column = torch.arange(self.width + 1, n + self.width - 1) % self.width
        mask = (column >= 1) & (column <= stabs)
        self.mask = nn.Buffer(mask.float()[None, None], persistent=False)

    def forward(self, x: torch.Tensor, repad: bool) -> torch.Tensor:
        n = self.rounds * self.width
        y = self.rows[0](x[:, :, 0:n])  # [1, out!, n - 2]
        y = y + self.rows[1](x[:, :, self.width : self.width + n])
        y = y + self.rows[2](x[:, :, 2 * self.width : 2 * self.width + n])
        y = torch.relu(y) * self.mask
        if not repad:
            return y
        z = y[:, :, : self.width + 1] * 0
        return torch.cat([z, y, z], dim=2)


class FiLMDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        num_stabilisers: int,
        rounds: int,
        channels: list[int] = [128, 256, 512],
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
        plus dense head is one full-length `Conv1d`. `load_reference` loads the
        weights of the plain `Conv2d` / `Linear` form.

        Args:
            num_stabilisers: Syndrome bits per round (d - 1)
            rounds:          Rounds r in the window
            channels:        Output channels of each block (the paper's 128, 256, 512)
        """
        super().__init__()
        self.rounds, self.stabs = rounds, num_stabilisers
        self.width = num_stabilisers + 2
        self.first = nn.Linear(9, channels[0])
        self.convs = nn.ModuleList(
            _Conv2d(i, o, rounds, num_stabilisers)
            for i, o in zip(channels[:-1], channels[1:])
        )
        # The head sees the padded grid, minus its first and last (padding) cell,
        # so a fifth of its weights (d=11) stay zero and unused.
        self.head = nn.Conv1d(
            channels[-1], num_stabilisers + 1, rounds * self.width - 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (1, rounds, num_stabilisers, 1): detection events

        Returns:
            Tensor of shape (1, 1, num_stabilisers + 1): data-qubit flip logits
        """
        xp = _pad(x)  # [1, T+2, S+2, 1!]
        taps = [
            xp[:, i : i + self.rounds, j : j + self.stabs]
            for i in range(3)
            for j in range(3)
        ]
        x = torch.relu(self.first(torch.cat(taps, dim=3)))  # [1, T, S, C!]
        x = _pad(x).reshape(1, (self.rounds + 2) * self.width, -1)  # [1, L, C!]
        x = x.transpose(1, 2)  # [1, C!, L]
        for i, conv in enumerate(self.convs):
            x = conv(x, repad=i < len(self.convs) - 1)  # [1, C!, L or L-2*width-2]
        return self.head(x).transpose(1, 2)  # [1, 1, d!]

    @torch.no_grad()
    def load_reference(self, convs: list[nn.Conv2d], head: nn.Linear):
        """Load the plain form: `Conv2d(., ., 3, padding=1)` blocks, then
        `Linear` over the flattened (channels, rounds, stabilisers) output."""
        self.first.weight.copy_(convs[0].weight.flatten(1))
        self.first.bias.copy_(convs[0].bias)
        for conv, ref in zip(self.convs, convs[1:]):
            for dt, row in enumerate(conv.rows):
                row.weight.copy_(ref.weight[:, :, dt])
            conv.rows[0].bias.copy_(ref.bias)
        w = head.weight.view(head.out_features, -1, self.rounds, self.stabs)
        self.head.weight.zero_()
        for t in range(self.rounds):
            at = t * self.width
            self.head.weight[:, :, at : at + self.stabs] = w[:, :, t]
        self.head.bias.copy_(head.bias)

    @torch.no_grad()
    def fold_film(self, gamma: list[torch.Tensor], beta: list[torch.Tensor]):
        """Bake per-block gamma, beta (each of shape (channels[i],)) into the convs."""
        first_and_rows = [self.first] + [c.rows[0] for c in self.convs]
        for layer, g, b in zip(first_and_rows, gamma, beta):
            layer.bias.mul_(g).add_(b)
        self.first.weight.mul_(gamma[0][:, None])
        for conv, g in zip(self.convs, gamma[1:]):
            for row in conv.rows:
                row.weight.mul_(g[:, None, None])


class StreamingFiLMDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        num_stabilisers: int,
        channels: list[int],
        kernel_size: int,
        dilation: int,
        film_input: bool = False,
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
        """
        super().__init__()
        self.num_stabilisers = num_stabilisers
        self.channels = channels
        self.film_input = film_input
        ins = [num_stabilisers] + channels[:-1]
        self.convs = nn.ModuleList(
            PaddedConv1d(i, o, kernel_size, dilation=dilation)
            for i, o in zip(ins, channels)
        )
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
        for conv, c in zip(self.convs, self.channels):
            x = conv(x)  # [B, c!, T]
            if self.film_input:
                gamma = film[:, offset : offset + c]
                beta = film[:, offset + c : offset + 2 * c]
                x = gamma * x + beta
                offset += 2 * c
            x = torch.relu(x)
        return self.head(x.transpose(1, 2))  # [B, T, d!]

    @torch.no_grad()
    def fold_film(self, gamma: list[torch.Tensor], beta: list[torch.Tensor]):
        """Bake per-block gamma, beta (each of shape (channels[i],)) into the convs."""
        for conv, g, b in zip(self.convs, gamma, beta):
            conv.conv.weight.mul_(g[:, None, None])
            conv.conv.bias.mul_(g).add_(b)


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
    from vollo_model_zoo.vm import vollo_info

    meta = dict(distance=num_stabilisers + 1, channels=str(channels))
    if rounds is not None:
        # The paper's decoder: one decision per window of `rounds` rounds.
        input = torch.randn(1, rounds, num_stabilisers, 1)
        model = FiLMDecoder(num_stabilisers, rounds, channels)
        time_axis = None
        meta |= dict(model="window", rounds=rounds)
    else:
        # With film_input, gamma and beta ride along as extra input channels.
        num_inputs = num_stabilisers + 2 * sum(channels) * film_input
        input = torch.randn(1, num_inputs, 8)
        model = StreamingFiLMDecoder(
            num_stabilisers, channels, kernel_size, dilation, film_input
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
    # after the final round is tens of us, and (d=11) its weights are 2.2M.
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
