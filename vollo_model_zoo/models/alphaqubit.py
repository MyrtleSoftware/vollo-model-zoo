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
    """
    The (x, y) positions of the d^2 - 1 stabilisers of a rotated surface code,
    scaled so that the code spans [-0.5, 0.5]. Returns [N, 2].
    """
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
    The paper's spatial RoPE, precomputed as one rotation matrix per stabiliser.

    Returns R: [N, dh, dh], where `t @ R[n]` equals the usual elementwise form
    `t * cos + rot(t) * sin` for stabiliser n. Within a head, the first half of
    the channels is rotated by the stabiliser's x coordinate and the second half
    by its y. Each half is laid out as `[a | b]`, which `rot` maps to `[-b | a]`.
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
        if self.rotation is not None:
            # RoPE as a matmul with each stabiliser's matrix:
            # [h n 1 dh!] @ [n dh dh]
            q = (q.unsqueeze(-2) @ self.rotation).squeeze(-2)
            k = (k.unsqueeze(-2) @ self.rotation).squeeze(-2)
        # The softmax is written out by hand, for three reasons.
        #
        # 1. exp(s) is computed as exp2(s / ln 2), which is cheaper on Vollo
        #    (see `_RGLRU`). The 1 / ln 2 is merged into the usual 1 / sqrt(dh)
        #    scale on q.
        # 2. Vollo has no max reduction, so the scores are not shifted by their
        #    maximum first. The result is the same, but in fp32 PyTorch exp
        #    overflows for scores above ~88.
        e = torch.exp2((q * (dh**-0.5 / math.log(2))) @ k.transpose(1, 2))  # [h n M!]
        # 3. The division by the sum comes after the value matmul, so the sum
        #    can run alongside the matmul. `torch.softmax(s) @ v` takes 2.5-5%
        #    more cycles on V80 and up to 17% more on IA-420f (SDK 29.1.0).
        a = (e @ v) / e.sum(-1, keepdim=True)  # [h n dh!]
        return self.proj_o(a.transpose(0, 1).reshape(-1, h * dh))


class _Layer(nn.Module):
    """
    The paper's transformer layer (Fig. S1a): attention, then a gated dense
    block. Each is pre-norm with a residual connection:

        x = x + attn(norm_attn(x))
        h = norm_dense(x)
        x = x + down(gelu(up(h)) * gate(h))

    For cross-attention, `context` is normalised with `norm_attn` too.
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
    Griffin's recurrent block (https://arxiv.org/abs/2402.19427), applied to
    each stabiliser independently. It follows the paper's Fig. S3, which has
    neither Griffin's Conv1D nor its MLP:

        h = norm(x)
        u = inp(h)
        y = x + out(state * gelu(gate(h)))

    `state` is updated by the RG-LRU, which is element-wise. With r and i the
    sigmoid recurrence and input gates:

        a = exp(-8 * softplus(decay) * r)
        state = a * state + sqrt(1 - a^2) * (i * u)

    Unlike a GRU, there is no matrix product inside the recurrence.

    Precision caveat: in bf16, `1 - a^2` rounds to 0 when
    `8 * softplus(decay) * r` is below about 1e-3, so the channels that decay
    most slowly ignore new input.
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
        # exp(y) is written as exp2(y / ln 2), because Vollo can fold a constant
        # scale into an exp2 but not into an exp. Doing this here and in
        # attention's softmax saves 1.6-1.9% of the cycles at d=5; d=3 changes
        # by less than 1% either way. Use exp2 in both places or in neither:
        # exp here with exp2 there costs 7% at d=3 on IA-420f (SDK 29.1.0).
        c = -8 / math.log(2)
        a = torch.exp2(c * torch.nn.functional.softplus(self.decay) * r)
        state = a * state + torch.sqrt(1 - a * a) * (i * u)
        return x + self.out(state * torch.nn.functional.gelu(self.gate(h))), state


class _Step(nn.Module):
    """
    One step: embed a group of `cycles` syndrome cycles, then apply an RG-LRU
    block, the transformer layers and a second RG-LRU block.
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
    ):
        super().__init__()
        # The paper embeds each cycle's bits, then concatenates the group's
        # embeddings and projects them. With no nonlinearity in between, that
        # is the same as a single Linear on all 2G bits.
        self.embed = nn.Linear(2 * cycles, dim)
        # The paper's position embedding is a Linear(2, C) of each stabiliser's
        # (x, y). Vollo can't compile a Linear whose input is a constant, so it
        # is evaluated once here and kept as a buffer, which is saved in the
        # state dict.
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
    The real-time AlphaQubit 2 decoder (AQ2-RT) for the surface code, from
    Senior et al. (https://arxiv.org/abs/2512.07737), Fig. 1b and §A.1.5. Its
    predecessor is AlphaQubit (https://arxiv.org/abs/2310.05900).

    Each of the N = d^2 - 1 stabilisers has a representation of `dim` channels.
    One step takes a group of `cycles` syndrome cycles (a measurement and a
    detection event per stabiliser per cycle) and:

    1. embeds them and adds the stabiliser's position embedding;
    2. applies a Griffin RG-LRU block;
    3. applies `layers` transformer layers across the stabilisers, with spatial
       RoPE and a gated dense block;
    4. applies a second RG-LRU block.

    The two RG-LRU states are the only memory between steps. There are no
    convolutions.

    The paper makes its prediction once, at the end of an experiment (see
    `AlphaQubitReadout`), so this streaming model outputs the stabilisers'
    representations at every step.

    Not modelled: at the end of an experiment the paper recomputes a final
    cycle from the data-qubit measurements, embeds it on its own and runs it
    through this same stack. That is one more step of the same cost, before
    the readout.

    For AQ2-RT the paper gives C=128, G=6 and the layer counts, but not the
    heads or the dense block. This file chooses:

    - 4 heads of 32 channels and a dense expansion of 4, as in the paper's
      larger models (Table S1);
    - a gated dense block, `down(gelu(up(h)) * gate(h))`, as in Griffin's MLP;
    - RoPE frequencies spaced geometrically from 20 down to about 0.36 in each
      half of a head, with coordinates in [-0.5, 0.5];
    - eps = 1e-5 in RMSNorm.
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
    AQ2-RT's readout, which runs once at the end of an experiment:

    1. two transformer layers across the stabilisers;
    2. a query is formed from the mean over the stabilisers plus a learned
       observable embedding;
    3. the query goes through two cross-attention layers over the stabilisers,
       then two residual dense layers;
    4. a Linear gives the logit for whether the logical qubit flipped.
    """

    @beartype
    def __init__(self, distance: int, dim: int = 128, heads: int = 4, expand: int = 4):
        super().__init__()
        n = distance**2 - 1
        rotation = _rope_rotation(distance, dim // heads)
        self.mix = nn.Sequential(
            *[_Layer(dim, heads, expand, rotation) for _ in range(2)]
        )
        # The mean over the stabilisers, as a matmul with a constant row of 1 / N
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
        allow_dynamic_weights=True,  # needed for attention's matmuls
        quick_compile=True,
        meta=dict(
            part=part,
            d=distance,
            **(dict(cycles=cycles, layers=layers) if part == "step" else {}),
        ),
    )


# Ordered by parameter count
SIZES = [
    # The paper's AQ2-RT for the surface code: C=128, two transformer layers
    # between the RG-LRU blocks, and G=6 syndrome cycles per step
    dict(part="step", distance=3, cycles=6, layers=2),
    dict(part="step", distance=5, cycles=6, layers=2),
    # Not in the paper, which uses G=6, so these would need retraining: G=16
    # cycles per step. A step costs the same as at G=6, so more cycles are
    # decoded per second, but the final decision can come up to 16 cycles later.
    dict(part="step", distance=3, cycles=16, layers=2),
    dict(part="step", distance=5, cycles=16, layers=2),
    # ~1M parameter baseline. Not a paper size: it is AQ2-RT with three
    # transformer layers, as in the paper's Willow model (whose C is 256), at
    # d=3 so that it fits every board. Parameters:
    #
    #   one transformer layer                                    264,064
    #     norms       2 * 128                                  =     256
    #     attention   4 * (128 * 128 + 128)                    =  66,048
    #     dense       2 * (128 * 512 + 512) + 512 * 128 + 128  = 197,760
    #   one RG-LRU block
    #     norm 128 + 5 * (128 * 128 + 128) + decay 128         =  82,816
    #   embed         12 * 128 + 128                           =   1,664
    #
    #   total         3 * 264,064 + 2 * 82,816 + 1,664         = 959,488
    dict(part="step", distance=3, cycles=6, layers=3),
    # The readout. The sizes above cost their latency at every step; this one
    # is paid once per experiment, after the last step.
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
