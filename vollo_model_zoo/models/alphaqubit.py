from collections.abc import Generator
from pathlib import Path
from typing import Literal, Optional

import torch
import vollo_torch
from beartype import beartype
from torch import nn


def _norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-5)


def _head_partitions(heads: int, cores: int) -> int:
    # The largest even grouping of the heads, one core per group: uneven groups
    # are far slower than none
    return max(p for p in range(1, min(heads, cores) + 1) if heads % p == 0)


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


def _rope_tables(distance: int, dim_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cos and sin of the spatial RoPE angles, [N, dim_head]. A head's first half
    of channels rotates with x, the second with y; each half is `[a | b]`, with
    each `(a_i, b_i)` rotated as a pair. The paper gives no frequencies: the
    geometric ladder from 20 down to about 0.36 (dim_head 32) is ours.
    """
    q = dim_head // 4
    freq = 20 * 10.0 ** (-2 * torch.arange(q) / q)
    x, y = (_stabiliser_coordinates(distance)[:, a : a + 1] * freq for a in (0, 1))
    angle = torch.cat([x, x, y, y], dim=1)
    return angle.cos(), angle.sin()


class _Attention(nn.Module):
    """
    Multi-head attention of `x` over `context` (default: itself). With
    `head_partitions`, head group `p` (its projections, scores and softmax) is
    pinned to core `p`; the output projection stays outside so it is not split.

    With `rope` (cos and sin tables), q and k are rotated as `t * cos + rot(t) *
    sin`, where `rot` maps each `[a | b]` channel pair to `[-b | a]`: a matmul
    with a signed permutation of a head's channels, so no slice or concatenate
    over the data dimension is needed.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        head_partitions: int,
        rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.heads_per_core = heads // head_partitions
        self.dim_head = dim_head
        width = self.heads_per_core * dim_head
        # q, k and v, one Linear per head group
        self.q, self.k, self.v = (
            nn.ModuleList(nn.Linear(dim, width) for _ in range(head_partitions))
            for _ in range(3)
        )
        self.rope = rope is not None
        if rope:
            cos, sin = rope
            self.cos = nn.Buffer(cos, persistent=False)  # [N dh]
            self.sin = nn.Buffer(sin, persistent=False)
            # rot as a matrix: t @ rot, t = [a | b] per half -> [-b | a]
            q, half = torch.arange(dim_head // 4), dim_head // 2
            rot = torch.zeros(dim_head, dim_head)
            for a in (q, q + half):  # the x half, then the y half
                rot[a + dim_head // 4, a] = -1
                rot[a, a + dim_head // 4] = 1
            self.rot = nn.Buffer(rot, persistent=False)
        self.proj_o = nn.Linear(heads * dim_head, dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [N, C!], context: [M, C!] -> [N, C!]"""
        context = x if context is None else context
        h, dh = self.heads_per_core, self.dim_head

        def heads(t):
            return t.view(-1, h, dh).transpose(0, 1)  # [h N dh!]

        def rotate(t):  # [h N dh!]
            return t * self.cos + (t @ self.rot) * self.sin  # [h N dh!] @ [dh dh]

        outs = []
        for p, (q, k, v) in enumerate(zip(self.q, self.k, self.v)):
            with vollo_torch.CorePartition([p]):
                q, k, v = heads(q(x)), heads(k(context)), heads(v(context))
                if self.rope:
                    q, k = rotate(q), rotate(k)
                s = (q * dh**-0.5) @ k.transpose(1, 2)  # [h N dh!] @ [h dh M!]
                a = torch.softmax(s, dim=-1) @ v  # [h N M!] @ [h M dh!]
                outs.append(a.transpose(0, 1).reshape(-1, h * dh))  # [N h*dh!]
        return self.proj_o(torch.cat(outs, dim=-1))


class _Layer(nn.Module):
    """
    Pre-norm transformer layer, as in the paper's Fig. S1a: attention, then a
    gated dense block, each behind its own RMSNorm and residual,
    `x = x + Attn(RMSNorm(x))`, then `x = x + down(gelu(up(h)) * gate(h))` with
    `h = RMSNorm(x)`. The gated form is Griffin's MLP block; the paper does not
    spell it out. For cross-attention the context goes through the first norm.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        expand: int,
        head_partitions: int,
        rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.norm_attn, self.norm_dense = _norm(dim), _norm(dim)
        self.attn = _Attention(dim, heads, 32, head_partitions, rope)
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
    block, per stabiliser, as drawn in the paper's Fig. S3 (no Conv1D, no MLP):
    `x + out(RG-LRU(inp(h)) * gelu(gate(h)))`, `h = RMSNorm(x)`. The RG-LRU is
    element-wise, `a = exp(-8 softplus(L) r)`, `state = a state + sqrt(1 - a^2) i u`,
    so unlike a GRU there is no matrix product inside the recurrence.
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
        a = torch.exp(-8 * torch.nn.functional.softplus(self.decay) * r)
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
        head_partitions: int,
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
        rope = _rope_tables(distance, 32)
        self.rnn_in, self.rnn_out = _RGLRU(dim), _RGLRU(dim)
        self.layers = nn.Sequential(
            *[
                _Layer(dim, heads, expand, head_partitions, rope=rope)
                for _ in range(layers)
            ]
        )

    def forward(self, syndromes: torch.Tensor, state: list[torch.Tensor]):
        """syndromes: [N, 2G!], state: two [N, C!] -> [N, C!], new state"""
        x = self.embed(syndromes) + self.position  # [N C!]
        x, rnn_in = self.rnn_in(x, state[0])
        x, rnn_out = self.rnn_out(self.layers(x), state[1])
        return x, [rnn_in, rnn_out]


class AlphaQubit(nn.Module):
    """
    The real-time AlphaQubit 2 decoder (AQ2-RT) for the surface code, after Senior
    et al. (https://arxiv.org/abs/2512.07737); its predecessor is AlphaQubit
    (https://arxiv.org/abs/2310.05900). Each of the N = d^2 - 1 stabilisers
    carries a `dim`-vector. A step consumes a group of `cycles` syndrome cycles
    (a measurement and a detection event each), embeds them with the stabiliser's
    position, and applies a Griffin RG-LRU block, `layers` pre-norm transformer
    layers across the stabilisers (spatial RoPE, gated dense) and a second RG-LRU
    block. Only the two RG-LRU states carry history; there are no convolutions.

    The paper reads out once per experiment, not per step (see
    `AlphaQubitReadout`), so this streamed model emits the stabilisers'
    representations. At the end of an experiment the paper also runs the final
    cycle, recomputed from the data-qubit measurements, through this same stack
    with its own embedding: one more step of the same cost before the readout.
    """

    @beartype
    def __init__(
        self,
        distance: int,
        cycles: int,
        dim: int,
        layers: int,
        heads: int,
        expand: int,
        head_partitions: int,
    ):
        super().__init__()
        self.scan = vollo_torch.nn.Scan(
            _Step(distance, cycles, dim, layers, heads, expand, head_partitions)
        )
        self.rnn_in_0 = nn.Buffer(torch.zeros(distance**2 - 1, dim), persistent=False)
        self.rnn_out_0 = nn.Buffer(torch.zeros(distance**2 - 1, dim), persistent=False)

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
    def __init__(
        self, distance: int, dim: int, heads: int, expand: int, head_partitions: int
    ):
        super().__init__()
        n = distance**2 - 1
        rope = _rope_tables(distance, 32)
        self.mix = nn.Sequential(
            *[_Layer(dim, heads, expand, head_partitions, rope=rope) for _ in range(2)]
        )
        self.mean = nn.Buffer(torch.full((1, n), 1 / n), persistent=False)
        self.observable = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.cross = nn.ModuleList(
            _Layer(dim, heads, expand, head_partitions) for _ in range(2)
        )
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
def _vm(
    variant: Literal["aq2", "readout"],
    distance: int,
    dim: int,
    layers: int,
    heads: int,
    expand: int,
    config: str,
    partitioned: bool = True,
    cycles: int = 6,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    n = distance**2 - 1
    partitions = _head_partitions(heads, CONFIGS[config].num_cores)
    partitions = partitions if partitioned else 1
    args = distance, dim, heads, expand, partitions
    match variant:
        case "aq2":
            model = AlphaQubit(distance, cycles, dim, layers, heads, expand, partitions)
            x, time_axis = torch.randn(2, n, 2 * cycles), 0
        case "readout":
            model = AlphaQubitReadout(*args)
            x, time_axis = torch.randn(n, dim), None
    return vollo_info(
        model,
        x,
        config=config,
        time_axis=time_axis,
        allow_dynamic_weights=True,
        quick_compile=True,
        meta=dict(
            model=variant,
            d=distance,
            **(dict(cycles=cycles) if variant == "aq2" else {}),
            dim=dim,
            layers=layers,
            heads=heads,
            partitions=partitions,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    aq2 = dict(variant="aq2", expand=4)
    # Ordered by parameter count, which is what is monotonic on every config
    for x in [
        # AQ2-RT: C=128, 1 RG-LRU + 2 transformer + 1 RG-LRU, G=6 cycles per step
        dict(**aq2, distance=3, dim=128, layers=2, heads=4),
        dict(**aq2, distance=5, dim=128, layers=2, heads=4),
        # What head partitioning buys: the same step on a single core
        dict(**aq2, distance=5, dim=128, layers=2, heads=4, partitioned=False),
        # BEYOND THE PAPER (G <= 6; needs retraining): G=16 cycles per step. A step
        # costs the same, so per-cycle throughput rises, but the final step waits
        # up to 16 cycles longer before the decision.
        dict(**aq2, distance=3, dim=128, layers=2, heads=4, cycles=16),
        dict(**aq2, distance=5, dim=128, layers=2, heads=4, cycles=16),
        # ~1M parameter baseline, not a paper size; d=3 so that it fits every board.
        # AQ2-RT's C=128, heads and widening, with the three transformer layers of
        # the paper's Willow model (whose C is 256). Per layer: two norms 2 * 128 +
        # attention (q, k, v and output projections: 4 * (128 * 128 + 128)) =
        # 66,048 + dense (up, gate, down: 2 * (128 * 512 + 512) + 512 * 128 + 128)
        # = 197,760 -> 264,064. Three layers 792,192; each RG-LRU block norm 128 +
        # 5 * (128 * 128 + 128) + decay 128 = 82,816, two 165,632; embed 12 * 128
        # + 128 = 1,664: 959,488.
        dict(**aq2, distance=3, dim=128, layers=3, heads=4),
        # AQ2-RT reads out once per experiment, not per step: the streamed entries
        # above are the per-step cost, this is the one-off cost after the last step
        dict(variant="readout", distance=5, dim=128, layers=2, heads=4, expand=4),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
