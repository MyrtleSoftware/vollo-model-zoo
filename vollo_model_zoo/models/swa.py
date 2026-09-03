from collections.abc import Generator
from pathlib import Path
from typing import Optional

import torch
import vollo_torch
from beartype import beartype
from torch import nn


class SlidingWindowAttention(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        window_size: int,
        bias: bool = True,
        mask: bool = True,
        late_norm: bool = True,
        head_partitions: Optional[int] = None,
    ):
        """
        A self-attention sublayer that is causal and windowed: each timestep
        attends to itself and the `window_size - 1` timesteps before it.

        Args:
            dim:             Input/output dimension
            heads:           Number of attention heads
            dim_head:        Dimension of each attention head
            window_size:     Number of timesteps a query attends over, itself
                             included; also the length of the K/V scan state
            bias:            Whether the query, key, value, and output projections
                             use biases.
            mask:            Whether to mask the window slots that have not been
                             filled yet, over the first `window_size - 1` timesteps
                             of a sequence. Costs a third scan state and a pointwise
                             add; turn it off if the accelerator is only ever read
                             after streaming in a warm-up sequence
            late_norm:       Whether to normalise the attention weights after the
                             value matmul rather than before it -- the same
                             arithmetic either way (see the comment at the
                             softmax), but it shortens the step's critical path.
                             Worth 3-4% of both latencies on a model partitioned
                             over its heads, which is what `head_partitions`
                             below is for. Unpartitioned it is a trade rather
                             than a win: still ~2-5% off `latency_spaced`, but up
                             to 22% onto `latency_contiguous` on a 6-core config,
                             so turn it off if you run one scan over all the
                             heads and stream back-to-back
            head_partitions: How many head groups to split the heads into,
                             group `p` pinned to core `p`. Must divide `heads`, and
                             must not exceed the cores in the target Vollo config.
        """
        super().__init__()

        if window_size < 2:
            raise ValueError(f"window_size ({window_size}) must be at least 2")

        self.mask = mask
        self.head_partitions = head_partitions

        if head_partitions is not None:
            if not 1 <= head_partitions <= heads:
                raise ValueError(
                    f"head_partitions ({head_partitions}) must be between 1 and "
                    f"the number of heads ({heads})"
                )
            if heads % head_partitions:
                raise ValueError(
                    f"heads ({heads}) must be a multiple of head_partitions "
                    f"({head_partitions}), or the groups are uneven and the "
                    f"busiest core sets the latency"
                )

        heads_per_scan = heads if head_partitions is None else heads // head_partitions

        if head_partitions is None:
            self.scan = vollo_torch.nn.Scan(
                _SlidingWindowAttentionStep(
                    dim, heads_per_scan, dim_head, bias, mask, late_norm
                )
            )
        else:
            self.scans = nn.ModuleList(
                vollo_torch.nn.Scan(
                    _SlidingWindowAttentionStep(
                        dim, heads_per_scan, dim_head, bias, mask, late_norm
                    )
                )
                for _ in range(head_partitions)
            )

        self.proj_o = nn.Linear(heads * dim_head, dim, bias=bias)

        # Same zeros initial buffer for each head partition

        self.k_0 = nn.Buffer(
            torch.zeros(heads_per_scan, window_size, dim_head), persistent=False
        )

        self.v_0 = nn.Buffer(
            torch.zeros(heads_per_scan, window_size, dim_head), persistent=False
        )

        if self.mask:
            self.bias_0 = nn.Buffer(
                torch.full((window_size,), float("-inf")), persistent=False
            )

        self.register_load_state_dict_pre_hook(self._load_any_partitioning)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Time, dim)

        Returns:
            x: Tensor of shape (Time, dim)
        """
        state = [self.k_0, self.v_0]

        if self.mask:
            state.append(self.bias_0)

        if self.head_partitions is None:
            return self.proj_o(self.scan(x, state))

        outs = []

        for p, scan in enumerate(self.scans):
            with vollo_torch.CorePartition([p]):
                outs.append(scan(x, state))

        return self.proj_o(torch.cat(outs, dim=-1))

    _PARTITIONED_KEYS = tuple(
        f"{name}.{suffix}"
        for name in ("proj_q", "proj_k", "proj_v")
        for suffix in ("weight", "bias")
    )

    def _load_any_partitioning(self, _module, state_dict, prefix, *_hook_args):
        """
        Let a checkpoint saved at one `head_partitions` load at any other.
        """
        for key in self._PARTITIONED_KEYS:
            wide = f"{prefix}scan.step.{key}"
            part = f"{prefix}scans.{{}}.step.{key}"

            # Gather.
            tensor = state_dict.pop(wide, None)

            if tensor is None:
                parts = []

                while part.format(len(parts)) in state_dict:
                    parts.append(state_dict.pop(part.format(len(parts))))

                # A partial set is merged too, so `load_state_dict` reports the
                # shortfall as a size mismatch rather than the layer silently
                # keeping untrained weights. Nothing at all is `bias=False`.
                if parts:
                    tensor = torch.cat(parts)

            if tensor is None:
                continue

            # Split.
            if self.head_partitions is None:
                state_dict[wide] = tensor
            else:
                for p, chunk in enumerate(tensor.chunk(self.head_partitions)):
                    state_dict[part.format(p)] = chunk


class _SlidingWindowAttentionStep(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        bias: bool,
        mask: bool,
        late_norm: bool,
    ):
        """
        The scan function, see SlidingWindowAttention for arguments.
        """
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.mask = mask
        self.late_norm = late_norm

        inner_dim = heads * dim_head

        self.proj_q = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_k = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_v = nn.Linear(dim, inner_dim, bias=bias)

        if mask:
            self.new_slot_bias = nn.Buffer(torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        """
        Args:
            x:     Tensor of shape (dim!), one timestep
            state: [K window, V window], both of shape (heads, window, dim_head!),
                   plus -- when masking warm-up -- the per-slot additive score
                   biases, of shape (window!)

        Returns:
            This group's attention output, of shape (inner!), and the updated
            state. The output projection is the caller's, so that partitioning
            does not split it
        """
        if self.mask:
            k_win, v_win, bias_win = state
        else:
            (k_win, v_win), bias_win = state, None

        q = self.proj_q(x).view(self.heads, self.dim_head)  # [h dh!]
        k = self.proj_k(x).view(self.heads, self.dim_head)  # [h dh!]
        v = self.proj_v(x).view(self.heads, self.dim_head)  # [h dh!]

        # Slide the windows: evict the oldest entry, append this timestep's.
        k_win = torch.cat([k_win[:, 1:, :], k.unsqueeze(1)], dim=1)  # [h w dh!]
        v_win = torch.cat([v_win[:, 1:, :], v.unsqueeze(1)], dim=1)  # [h w dh!]
        if bias_win is not None:
            bias_win = torch.cat([bias_win[1:], self.new_slot_bias])  # [w!]

        # Free: the compiler folds a constant scale into `proj_q`, so this
        # emits no instruction. But keep it *below* the window slides -- the
        # emission order decides the schedule, and hoisting it up to `proj_q`
        # measures 1% slower for an identical program.
        q = q * self.scale

        # [h 1 dh!] @ [h dh! w] -> [h 1 w!]
        scores = q.unsqueeze(1) @ k_win.transpose(1, 2)

        if bias_win is not None:
            # The bias will mask the unfilled windows from the softmax
            scores = scores + bias_win  # [h 1 w!]

        if self.late_norm:
            # The softmax written out, so that its division can move to the far
            # side of the value matmul (see `late_norm`):
            #
            #     softmax(s) @ V == (exp(s) @ V) / sum(exp(s))
            #
            # The sum then runs alongside the value matmul instead of ahead of
            # it, which is what shortens the step's critical path. Nothing is
            # lost by not calling `torch.softmax`: Vollo has no reduce-max, so
            # it compiles to this same unshifted exp/sum/divide anyway --
            # writing it out is only what makes the reordering expressible.
            e = torch.exp(scores)  # [h 1 w!]

            # [h 1 w!] @ [h w dh!] -> [h 1 dh!], the divide broadcast over dh
            out = (e @ v_win) / e.sum(-1, keepdim=True)  # [h 1 dh!]
        else:
            attn = torch.softmax(scores, dim=-1)  # [h 1 w!]

            # [h 1 w!] @ [h w dh!] -> [h 1 dh!]
            out = attn @ v_win  # [h 1 dh!]

        # [h 1 dh!] -> [h dh!] -> [inner!]
        out = out.squeeze(1).reshape(self.heads * self.dim_head)

        new_state = [k_win, v_win]

        if bias_win is not None:
            new_state.append(bias_win)

        return out, new_state


class SlidingWindowBlock(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        window_size: int,
        bias: bool = True,
        mask: bool = True,
        late_norm: bool = True,
        head_partitions: Optional[int] = None,
        expand: float = 2.0,
    ):
        """
        A sliding window transformer block, given input `x`:

        ```math
        y1 <- rms-norm(x)
        y2 <- SWA(y1)
        y3 <- x + y2

        y4 <- rms-norm(y3)
        y5 <- FFN(y4)
        y6 <- y3 + y5
        ```

        The FFN uses an swiglu activation with expansion size of `expand`. Both
        sublayers are pre-norm and residual,

        Args:
            expand: FFN hidden dimension as a multiple of `dim`; every other
                    argument is `SlidingWindowAttention`'s.
        """
        super().__init__()

        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attn = SlidingWindowAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            window_size=window_size,
            bias=bias,
            mask=mask,
            late_norm=late_norm,
            head_partitions=head_partitions,
        )

        hidden = int(dim * expand)

        self.ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.ffn_1 = nn.Linear(dim, hidden, bias=bias)
        self.ffn_2 = nn.Linear(dim, hidden, bias=bias)
        self.ffn_3 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Time, dim)

        Returns:
            x: Tensor of shape (Time, dim)
        """
        x = x + self.attn(self.attn_norm(x))

        h = self.ffn_norm(x)
        h = self.ffn_3(torch.nn.functional.silu(self.ffn_1(h)) * self.ffn_2(h))
        return x + h


@beartype
def _vm(
    dim: int,
    window_size: int,
    layers: int,
    config: str,
    mask: bool,
    late_norm: bool,
    expand: float,
    heads: int = 6,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    input = torch.randn(2, dim)

    # Try to partition across all cores
    if heads % (cores := CONFIGS[config].num_cores) == 0:
        head_partitions = cores
    else:
        head_partitions = None

    model = nn.Sequential().extend(
        SlidingWindowBlock(
            dim=dim,
            heads=heads,
            dim_head=32,
            window_size=window_size,
            mask=mask,
            late_norm=late_norm,
            head_partitions=head_partitions,
            expand=expand,
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
            masked=mask,
            late_norm=late_norm,
            partitions=head_partitions if head_partitions is not None else 0,
            dim=dim,
            window=window_size,
            layers=layers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, window_size=16, layers=1, mask=True, late_norm=True),
        dict(dim=32 * 6, window_size=32, layers=1, mask=True, late_norm=True),
        dict(dim=32 * 6, window_size=64, layers=1, mask=True, late_norm=True),
        # ~1M parameter baseline
        dict(dim=32 * 7, window_size=32, layers=2, mask=True, late_norm=True),
        dict(dim=32 * 7, window_size=32, layers=2, mask=False, late_norm=True),
        # The baseline again with the softmax normalised before the value
        # matmul instead of after it. Paired so the critical path `late_norm`
        # shortens is visible as a number: it is a win on both latencies for
        # every config here, since all of them partition these six heads over
        # their cores. Run one scan over all six and the sign of the
        # `contiguous` half flips -- see the `late_norm` docstring.
        dict(dim=32 * 7, window_size=32, layers=2, mask=True, late_norm=False),
        # big
        dict(dim=32 * 12, window_size=32, layers=6, mask=True, late_norm=True),
    ]:
        yield _vm(**x, expand=2.0, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
