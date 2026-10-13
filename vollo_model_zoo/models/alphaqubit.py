import math
from collections.abc import Callable, Generator
from contextlib import nullcontext
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


def _spread(
    heads: int, tokens: int, cores: int, weights: float, store: int
) -> tuple[int, int]:
    """
    How to spread a step over `cores` cores, as (token groups, head groups).

    Splitting only the heads leaves cores idle during attention (two of six at
    d=5), so the tokens are split into three or more groups as well when either:
    - there are at least two tokens per core, or
    - one core's weight store (`store` blocks) cannot hold all the weights
      (`weights`, in blocks). The compiler spreads the tokens itself by copying
      the weights to every core, which then does not work.
    Each token group holds a copy of the weights on its cores, so their share
    must fit. d=5 on six cores is three groups of eight tokens on two cores
    each: 10% faster on V80 and 20% on IA-420f and IA-840f (SDK 29.0.0).
    Otherwise only the heads are split, as at d=3 on V80, where a group's work
    is too short to share out.
    """
    head_groups = _head_partitions(heads, max(cores // 3, 1))
    split = tokens >= 2 * cores or weights > store
    if split and weights <= head_groups * store:
        return cores // head_groups, head_groups
    return 1, _head_partitions(heads, cores)


def _cat(ts: list[torch.Tensor], dim: int) -> torch.Tensor:
    return ts[0] if len(ts) == 1 else torch.cat(ts, dim=dim)


class _Cores:
    """
    How a step is spread over the cores: the N tokens (stabilisers) in
    `token_groups` contiguous groups, the smaller ones first, each on
    `head_groups` cores of its own. A group's per-token work (embedding, norms,
    RG-LRU, dense block, output projection) is pinned to its cores, and head
    group p of its attention to the p-th of them. Only attention mixes tokens,
    so the groups meet only there, to gather each other's keys and values: a
    concatenation along the token axis, not the data dimension. With one token
    group the per-token work is left to the compiler, which spreads the tokens
    itself, and with one head group as well nothing is pinned.
    """

    @beartype
    def __init__(self, tokens: int, token_groups: int = 1, head_groups: int = 1):
        if not 1 <= token_groups <= tokens:
            raise ValueError(f"{token_groups} token groups for {tokens} tokens")
        # The smaller groups first (8 tokens in 3: 2, 3, 3), about 1% faster
        # than last on IA-420f and IA-840f (SDK 29.0.0)
        bounds = [tokens * g // token_groups for g in range(token_groups + 1)]
        self.groups = list(zip(bounds, bounds[1:]))
        self.head_groups = head_groups

    def pin(self, g: int, p: Optional[int] = None):
        """Pin to token group `g`'s cores, or to the core of its head group `p`"""
        h = self.head_groups
        if len(self.groups) == 1 and (p is None or h == 1):
            return nullcontext()
        cores = range(g * h, (g + 1) * h) if p is None else [g * h + p]
        return vollo_torch.CorePartition(list(cores))

    def per_token(self, fn: Callable, *xs: torch.Tensor):
        """`fn` of each token group of `xs` (each [N, ...]) on its cores, concatenated"""
        if len(self.groups) == 1:
            return fn(*xs)
        outs = []
        for g, (a, b) in enumerate(self.groups):
            with self.pin(g):
                outs.append(fn(*(x[a:b] for x in xs)))
        if isinstance(outs[0], tuple):
            return tuple(_cat(list(o), dim=0) for o in zip(*outs))
        return _cat(outs, dim=0)


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
    Multi-head attention of `x` over `context` (default: itself), spread over
    the cores by `cores`: head group `p` of token group `g` (its projections,
    scores and softmax) on one core. The output projection is not split by
    heads: it runs per token group, after a concatenation of the heads. Each
    token group emits its keys and values before its queries, since every group
    waits for the keys and values while the queries stay at home (up to 3%
    faster, SDK 29.0.0).

    With `rope` (cos and sin tables), q and k are rotated as `t * cos + rot(t) *
    sin`, where `rot` maps each `[a | b]` channel pair to `[-b | a]`. Both terms
    are one matmul per stabiliser, `t @ (diag(cos) + rot diag(sin))`, with a
    constant matrix of two nonzeros per row: no slice or concatenate over the
    data dimension, and one instruction on the way to the scores rather than
    four (SDK 29.0.0).

    The softmax is normalised after the value matmul, `(exp(s) @ v) / sum(exp(s))`,
    so the sum runs alongside the matmul rather than ahead of it, and its
    exponential is `exp(s) = exp2(s / ln 2)`, the `1 / ln 2` joining q's `1 /
    sqrt(dh)` scale: an exp of a matmul's output costs an instruction more than
    an exp2 (SDK 29.0.0). It is unshifted (no `- max(s)`), as Vollo has no max
    reduction: equal to `torch.softmax` in exact arithmetic, but in PyTorch fp32
    `exp` overflows for scores above about 88, so mind that if you train with
    this file.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        cores: _Cores,
        rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.cores = cores
        self.heads_per_core = heads // cores.head_groups
        self.dim_head = dim_head
        width = self.heads_per_core * dim_head
        # q, k and v, one Linear per head group
        self.q, self.k, self.v = (
            nn.ModuleList(nn.Linear(dim, width) for _ in range(cores.head_groups))
            for _ in range(3)
        )
        self.rope = rope is not None
        if rope:
            # rot as a matrix: t @ rot, t = [a | b] per half -> [-b | a]
            q, half = torch.arange(dim_head // 4), dim_head // 2
            rot = torch.zeros(dim_head, dim_head)
            for a in (q, q + half):  # the x half, then the y half
                rot[a + dim_head // 4, a] = -1
                rot[a, a + dim_head // 4] = 1
            cos, sin = rope  # [N dh]
            rotation = torch.diag_embed(cos) + rot * sin.unsqueeze(1)  # [N dh dh]
            self.rotation = nn.Buffer(rotation, persistent=False)
        self.proj_o = nn.Linear(heads * dim_head, dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [N, C!], context: [M, C!] -> [N, C!]"""
        groups, pin = self.cores.groups, self.cores.pin
        if context is not None and len(groups) > 1:
            raise ValueError("cross-attention takes one token group")
        context = x if context is None else context
        h, dh = self.heads_per_core, self.dim_head

        def take(t, g):  # token group g's rows of t
            return t if len(groups) == 1 else t[groups[g][0] : groups[g][1]]

        def heads(t):  # [n h*dh!] -> [h n dh!]
            return t.view(-1, h, dh).transpose(0, 1)

        def rotate(t, g):  # [h n dh!] -> [h n 1 dh!] @ [n dh dh]
            return (t.unsqueeze(-2) @ take(self.rotation, g)).squeeze(-2)

        outs = [[] for _ in groups]  # [g][p]: [n h*dh!]
        for p, (q, k, v) in enumerate(zip(self.q, self.k, self.v)):
            queries, keys, values = [], [], []
            for g in range(len(groups)):
                with pin(g, p):
                    k_g = k(take(context, g))
                    values.append(heads(v(take(context, g))))
                    q_g = q(take(x, g))
                    k_g, q_g = heads(k_g), heads(q_g)
                    if self.rope:
                        k_g, q_g = rotate(k_g, g), rotate(q_g, g)
                    keys.append(k_g)
                    queries.append(q_g)
            k, v = _cat(keys, dim=1), _cat(values, dim=1)  # [h M dh!]
            for g, q_g in enumerate(queries):
                with pin(g, p):
                    # The scores in base 2, s / ln 2: [h n dh!] @ [h dh M!]
                    e = torch.exp2((q_g * (dh**-0.5 / math.log(2))) @ k.transpose(1, 2))
                    a = (e @ v) / e.sum(-1, keepdim=True)  # [h n M!] @ [h M dh!]
                    outs[g].append(a.transpose(0, 1).reshape(-1, h * dh))
        ys = []
        for g, heads_out in enumerate(outs):
            with pin(g):
                ys.append(self.proj_o(_cat(heads_out, dim=-1)))  # [n C!]
        return _cat(ys, dim=0)


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
        cores: _Cores,
        rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.cores = cores
        self.norm_attn, self.norm_dense = _norm(dim), _norm(dim)
        self.attn = _Attention(dim, heads, 32, cores, rope)
        self.up = nn.Linear(dim, expand * dim)
        self.gate = nn.Linear(dim, expand * dim)
        self.down = nn.Linear(expand * dim, dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x: [N, C!], context (for cross-attention): [M, C!] -> [N, C!]"""
        each = self.cores.per_token
        h = each(self.norm_attn, x)
        a = self.attn(h, None if context is None else self.norm_attn(context))
        return each(self._dense, each(torch.add, x, a))

    def _dense(self, x: torch.Tensor) -> torch.Tensor:
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
        # exp(x) as exp2(x / ln 2): Vollo folds a constant scale into exp2, not
        # into exp (SDK 29.0.0)
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
        cores: _Cores,
    ):
        super().__init__()
        self.cores = cores
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
            *[_Layer(dim, heads, expand, cores, rope) for _ in range(layers)]
        )

    def forward(self, syndromes: torch.Tensor, state: list[torch.Tensor]):
        """syndromes: [N, 2G!], state: two [N, C!] -> [N, C!], new state"""
        each = self.cores.per_token

        def first(syndromes, position, state):  # [n 2G!], [n C!], [n C!]
            return self.rnn_in(self.embed(syndromes) + position, state)

        x, rnn_in = each(first, syndromes, self.position, state[0])
        x, rnn_out = each(self.rnn_out, self.layers(x), state[1])
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

    The step is spread over `token_groups` x `head_partitions` cores (see
    `_Cores`). Every spread computes the same function.
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
        token_groups: int = 1,
    ):
        super().__init__()
        cores = _Cores(distance**2 - 1, token_groups, head_partitions)
        self.scan = vollo_torch.nn.Scan(
            _Step(distance, cycles, dim, layers, heads, expand, cores)
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
    layers and a projection to the logical-flip logit. The spatial layers are
    spread like `AlphaQubit`'s step.
    """

    @beartype
    def __init__(
        self,
        distance: int,
        dim: int,
        heads: int,
        expand: int,
        head_partitions: int,
        token_groups: int = 1,
    ):
        super().__init__()
        n = distance**2 - 1
        rope = _rope_tables(distance, 32)
        cores = _Cores(n, token_groups, head_partitions)
        self.mix = nn.Sequential(
            *[_Layer(dim, heads, expand, cores, rope) for _ in range(2)]
        )
        self.mean = nn.Buffer(torch.full((1, n), 1 / n), persistent=False)
        self.observable = nn.Parameter(torch.randn(1, dim) * 0.02)
        # One query: its work is the context's keys and values, which the
        # compiler spreads over every core by tokens if its heads are not pinned
        self.cross = nn.ModuleList(
            _Layer(dim, heads, expand, _Cores(n)) for _ in range(2)
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

    conf, n = CONFIGS[config], distance**2 - 1

    def build(partitions: int, groups: int):
        if variant == "aq2":
            args = distance, cycles, dim, layers, heads, expand, partitions, groups
            return AlphaQubit(*args)
        return AlphaQubitReadout(distance, dim, heads, expand, partitions, groups)

    # What a token group holds: the step's weights, or the readout's spatial
    # layers' (its cross-attention is never split)
    held = build(1, 1) if variant == "aq2" else build(1, 1).mix
    weights = sum(p.numel() for p in held.parameters()) / conf.block_size**2
    groups = partitions = 1
    if partitioned:
        store = conf.weight_store_depth
        groups, partitions = _spread(heads, n, conf.num_cores, weights, store)
    model = build(partitions, groups)
    match variant:
        case "aq2":
            x, time_axis = torch.randn(2, n, 2 * cycles), 0
        case "readout":
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
            token_groups=groups,
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
        # What partitioning buys: the same step with no `CorePartition`, spread
        # over the cores by the compiler alone
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
