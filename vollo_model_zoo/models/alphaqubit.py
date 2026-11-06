import math
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Optional

import torch
import vollo_torch
from beartype import beartype
from torch import nn


def _norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-5)


@beartype
def _stabiliser_coordinates(distance: int) -> torch.Tensor:
    """[N, 2] (x, y) of a rotated surface code's d^2 - 1 stabilisers, edges at -0.5 / 0.5."""
    d = distance
    sites = [
        (x, y)
        for x in range(d + 1)
        for y in range(d + 1)
        if (1 <= x < d and 1 <= y < d)  # weight-4 plaquettes
        or (y in (0, d) and 1 <= x < d and (x + y) % 2 == 1)  # weight-2 edges
        or (x in (0, d) and 1 <= y < d and (x + y) % 2 == 0)
    ]
    return torch.tensor(sites, dtype=torch.float32) / d - 0.5


@beartype
def _rope_rotation(distance: int, dim_head: int) -> torch.Tensor:
    """
    The paper's spatial RoPE as one matrix per stabiliser, [N, dh, dh], so that
    `t @ R[n] = t * cos + rot(t) * sin` for stabiliser n. A head's first half of
    channels rotates with x, the second with y; each half is `[a | b]`, and `rot`
    maps it to `[-b | a]`. One matmul rather than the elementwise form with
    `rot` as slices: without, d=3 +13% cycles on V80 and +6% on IA-420f, d=5
    +16% and +12%, and IA-840f no longer fits d=5 or the readout (SDK 29.1.0).
    """
    q = dim_head // 4
    freq = 20 * 10.0 ** (-2 * torch.arange(q) / q)
    x, y = (_stabiliser_coordinates(distance)[:, a : a + 1] * freq for a in (0, 1))
    angle = torch.cat([x, x, y, y], dim=1)  # [N dh]
    rot = torch.zeros(dim_head, dim_head)
    for a in (torch.arange(q), torch.arange(q) + 2 * q):  # the x half, the y half
        rot[a + q, a], rot[a, a + q] = -1, 1
    return torch.diag_embed(angle.cos()) + rot * angle.sin().unsqueeze(1)


class _Attention(nn.Module):
    """
    Multi-head attention of `x` over `context` (default: itself), with the
    paper's spatial RoPE on q and k when given `rotation`.
    """

    @beartype
    def __init__(self, dim: int, heads: int, rotation: Optional[torch.Tensor] = None):
        super().__init__()
        self.heads, self.dim_head = heads, dim // heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj_o = nn.Linear(dim, dim)
        self.register_buffer("rotation", rotation, persistent=False)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [N, C!], context: [M, C!] -> [N, C!]"""
        context = x if context is None else context
        h, dh = self.heads, self.dim_head

        def split(t):  # [n h*dh!] -> [h n dh!]
            return t.view(-1, h, dh).transpose(0, 1)

        q, k, v = split(self.q(x)), split(self.k(context)), split(self.v(context))
        if self.rotation is not None:  # RoPE, one matmul: [h n 1 dh!] @ [n dh dh]
            q = (q.unsqueeze(-2) @ self.rotation).squeeze(-2)
            k = (k.unsqueeze(-2) @ self.rotation).squeeze(-2)
        # exp(s) as exp2(s / ln 2), the 1 / ln 2 joining q's 1 / sqrt(dh) (see
        # `_RGLRU`). Vollo has no max reduction, so the softmax is unshifted: in
        # PyTorch fp32, exp overflows for scores above ~88.
        e = torch.exp2((q * (dh**-0.5 / math.log(2))) @ k.transpose(1, 2))  # [h n M!]
        # Normalised after the value matmul, so the sum runs alongside it.
        # Against `torch.softmax(s) @ v`, both with exp: d=3 +2.5% cycles on
        # V80 and +17% on IA-420f, d=5 +5.3% and +0.1% (SDK 29.1.0).
        a = (e @ v) / e.sum(-1, keepdim=True)  # [h n dh!]
        return self.proj_o(a.transpose(0, 1).reshape(-1, h * dh))


class _Layer(nn.Module):
    """
    The paper's transformer layer (Fig. S1a), serial and pre-norm: attention,
    then a gated dense block, each behind its own RMSNorm and residual,
    `x = x + Attn(RMSNorm(x))`, then `x = x + down(gelu(up(h)) * gate(h))` with
    `h = RMSNorm(x)`. For cross-attention the context goes through the first
    norm.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        expand: int,
        rotation: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.norm_attn, self.norm_dense = _norm(dim), _norm(dim)
        self.attn = _Attention(dim, heads, rotation)
        self.up = nn.Linear(dim, expand * dim)
        self.gate = nn.Linear(dim, expand * dim)
        self.down = nn.Linear(expand * dim, dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [N, C!], context (for cross-attention): [M, C!] -> [N, C!]"""
        h = self.norm_attn(x)
        x = x + self.attn(h, None if context is None else self.norm_attn(context))
        h = self.norm_dense(x)
        return x + self.down(torch.nn.functional.gelu(self.up(h)) * self.gate(h))


class _RGLRU(nn.Module):
    """
    Griffin's recurrent block (https://arxiv.org/abs/2402.19427) in its residual
    block, per stabiliser, as in the paper's Fig. S3 (no Conv1D, no MLP):
    `x + out(RG-LRU(inp(h)) * gelu(gate(h)))`, `h = RMSNorm(x)`. The RG-LRU is
    element-wise, `a = exp(-8 softplus(L) r)`, `state = a state + sqrt(1 - a^2) i u`,
    so unlike a GRU there is no matrix product inside the recurrence.

    In bf16, `1 - a^2` rounds to 0 when `8 softplus(L) r` is below about 1e-3,
    so the slowest channels drop their input.
    """

    @beartype
    def __init__(self, dim: int):
        super().__init__()
        self.norm = _norm(dim)
        self.inp, self.gate, self.out = (nn.Linear(dim, dim) for _ in range(3))
        self.proj_r, self.proj_i = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.decay = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """x: [N, C!], state: [N, C!] -> y [N, C!], new state [N, C!]"""
        h = self.norm(x)
        u = self.inp(h)
        r, i = torch.sigmoid(self.proj_r(u)), torch.sigmoid(self.proj_i(u))
        # exp(y) as exp2(y / ln 2): Vollo folds a constant scale into an exp2,
        # not an exp. Here and in attention's softmax, without: d=5 +1.6% cycles
        # on V80 and +1.9% on IA-420f, d=3 +0.9% and -0.9%. Use exp2 in both or
        # in neither: exp here, exp2 there costs IA-420f d=3 7% (SDK 29.1.0).
        c = -8 / math.log(2)
        a = torch.exp2(c * torch.nn.functional.softplus(self.decay) * r)
        state = a * state + torch.sqrt(1 - a * a) * (i * u)
        return x + self.out(state * torch.nn.functional.gelu(self.gate(h))), state


class _Step(nn.Module):
    """One group of `cycles` syndrome cycles: embed, RG-LRU, transformers, RG-LRU."""

    @beartype
    def __init__(
        self,
        distance: int,
        cycles: int,
        dim: int,
        layers: int,
        heads: int,
        expand: int,
    ):
        super().__init__()
        # The paper embeds each cycle's bits and "concatenates and projects" the
        # group: two Linears with nothing between, i.e. one Linear on the 2G bits.
        self.embed = nn.Linear(2 * cycles, dim)
        # The paper's Linear(2, C) of the stabilisers' (x, y), evaluated once since
        # a Linear of a constant does not compile; saved in the state dict.
        self.position = nn.Buffer(
            nn.Linear(2, dim)(_stabiliser_coordinates(distance)).detach()
        )
        rotation = _rope_rotation(distance, dim // heads)
        self.rnn_in, self.rnn_out = _RGLRU(dim), _RGLRU(dim)
        self.layers = nn.Sequential(
            *[_Layer(dim, heads, expand, rotation) for _ in range(layers)]
        )

    def forward(self, syndromes: torch.Tensor, state: list[torch.Tensor]):
        """syndromes: [N, 2G!], state: two [N, C!] -> [N, C!], new state"""
        x, rnn_in = self.rnn_in(self.embed(syndromes) + self.position, state[0])
        x, rnn_out = self.rnn_out(self.layers(x), state[1])
        return x, [rnn_in, rnn_out]


class AlphaQubit(nn.Module):
    """
    The real-time AlphaQubit 2 decoder (AQ2-RT) for the surface code, after
    Senior et al. (https://arxiv.org/abs/2512.07737), Fig. 1b and §A.1.5; its
    predecessor is AlphaQubit (https://arxiv.org/abs/2310.05900). Each of the
    N = d^2 - 1 stabilisers carries a `dim`-vector. A step consumes a group of
    `cycles` syndrome cycles (a measurement and a detection event each), embeds
    them with the stabiliser's position, and applies a Griffin RG-LRU block,
    `layers` transformer layers across the stabilisers (spatial RoPE, gated
    dense) and a second RG-LRU block. Only the two RG-LRU states carry history;
    there are no convolutions.

    The paper reads out once per experiment, not per step (`AlphaQubitReadout`),
    so this streamed model emits the stabilisers' representations. At the end
    of an experiment the paper also embeds the final cycle, recomputed from the
    data-qubit measurements, by itself and runs it through this same stack: one
    more step of the same cost before the readout, not modelled here.

    Where the paper is silent, this file chooses (for AQ2-RT it gives C=128,
    G=6 and the layer counts, but not the heads or the dense block):
    - heads of 32 (four at C=128) and a dense widening of 4, as in the paper's
      larger models (Table S1);
    - the dense block's gated form, `down(gelu(up(h)) * gate(h))`, Griffin's MLP;
    - RoPE frequencies: a geometric ladder from 20 down to about 0.36 per half
      head, over coordinates in [-0.5, 0.5];
    - RMSNorm's eps of 1e-5.
    """

    @beartype
    def __init__(
        self,
        distance: int,
        cycles: int = 6,
        dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        expand: int = 4,
    ):
        super().__init__()
        n = distance**2 - 1
        self.scan = vollo_torch.nn.Scan(
            _Step(distance, cycles, dim, layers, heads, expand)
        )
        self.rnn_in_0 = nn.Buffer(torch.zeros(n, dim), persistent=False)
        self.rnn_out_0 = nn.Buffer(torch.zeros(n, dim), persistent=False)

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """syndromes: [T, N, 2G!] -> stabiliser representations [T, N, C!]"""
        state = [self.rnn_in_0, self.rnn_out_0]
        return self.scan(syndromes, state, input_axis=0, output_axis=0)


class AlphaQubitReadout(nn.Module):
    """
    AQ2-RT's end-of-experiment readout: two spatial transformer layers, then a
    query (the mean over stabilisers plus a learned observable embedding) goes
    through two cross-attention layers over the stabilisers, two residual dense
    layers and a projection to the logical-flip logit.
    """

    @beartype
    def __init__(self, distance: int, dim: int = 128, heads: int = 4, expand: int = 4):
        super().__init__()
        n = distance**2 - 1
        rotation = _rope_rotation(distance, dim // heads)
        self.mix = nn.Sequential(
            *[_Layer(dim, heads, expand, rotation) for _ in range(2)]
        )
        self.mean = nn.Buffer(torch.full((1, n), 1 / n), persistent=False)
        self.observable = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.cross = nn.ModuleList(_Layer(dim, heads, expand) for _ in range(2))
        self.dense = nn.ModuleList(
            nn.Sequential(
                _norm(dim),
                nn.Linear(dim, expand * dim),
                nn.GELU(),
                nn.Linear(expand * dim, dim),
            )
            for _ in range(2)
        )
        self.logit = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C!] -> logit [1]"""
        x = self.mix(x)
        query = self.mean @ x + self.observable  # [1 N] @ [N C!] -> [1 C!]
        for layer in self.cross:
            query = layer(query, x)
        for dense in self.dense:
            query = query + dense(query)
        return self.logit(query).squeeze(0)


@beartype
def instance(
    part: Literal["step", "readout"],
    distance: int,
    config: str,
    cycles: int = 6,
    layers: int = 2,
) -> dict:
    n = distance**2 - 1
    match part:
        case "step":
            model = AlphaQubit(distance, cycles, layers=layers)
            x, time_axis = torch.randn(2, n, 2 * cycles), 0
        case "readout":
            model = AlphaQubitReadout(distance)
            x, time_axis = torch.randn(n, 128), None
    return dict(
        model=model,
        x=x,
        time_axis=time_axis,
        allow_dynamic_weights=True,  # attention's matmuls
        quick_compile=True,
        meta=dict(
            part=part,
            d=distance,
            **(dict(cycles=cycles, layers=layers) if part == "step" else {}),
        ),
    )


# Ordered by parameter count, which is monotonic on every config
SIZES = [
    # AQ2-RT for the surface code (C=128; 1 RG-LRU + 2 transformer + 1 RG-LRU),
    # one step of G=6 syndrome cycles
    dict(part="step", distance=3, cycles=6, layers=2),
    dict(part="step", distance=5, cycles=6, layers=2),
    # BEYOND THE PAPER (it uses G=6; needs retraining): G=16 cycles per step. A
    # step costs the same, so per-cycle throughput rises, but the final step
    # waits up to 16 cycles longer before the decision.
    dict(part="step", distance=3, cycles=16, layers=2),
    dict(part="step", distance=5, cycles=16, layers=2),
    # ~1M parameter baseline, not a paper size; d=3 so that it fits every board.
    # AQ2-RT's C=128, heads and widening, with the three transformer layers of
    # the paper's Willow model (whose C is 256). Per layer: two norms 2 * 128 +
    # attention (q, k, v and output projections: 4 * (128 * 128 + 128)) =
    # 66,048 + dense (up, gate, down: 2 * (128 * 512 + 512) + 512 * 128 + 128)
    # = 197,760 -> 264,064. Three layers 792,192; each RG-LRU block norm 128 +
    # 5 * (128 * 128 + 128) + decay 128 = 82,816, two 165,632; embed 12 * 128
    # + 128 = 1,664: 959,488.
    dict(part="step", distance=3, cycles=6, layers=3),
    # AQ2-RT reads out once per experiment, not per step: the entries above are
    # the per-step cost, this is the one-off cost after the last step
    dict(part="readout", distance=5),
]


@beartype
def main(config: str = "V80") -> Generator:
    from vollo_model_zoo.vm import sweep, vollo_info

    for size in sweep(SIZES, config):
        yield vollo_info(**instance(**size, config=config), config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
