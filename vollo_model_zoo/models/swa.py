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
        head_partitions: Optional[int] = None,
    ):
        """
        A self-attention sublayer that is causal and windowed: each timestep
        attends to itself and the `window_size - 1` timesteps before it.

        Args:
            dim:         Input/output dimension
            heads:       Number of attention heads
            dim_head:    Dimension of each attention head
            window_size: Number of timesteps a query attends over, itself
                         included; also the length of the K/V scan state
            bias:        Whether the query, key and value projections use
                         biases.
            mask: Whether to mask the window slots that have not been
                         filled yet, over the first `window_size - 1` timesteps
                         of a sequence. Costs a third scan state and a pointwise
                         add; turn it off if the accelerator is only ever read
                         after streaming in a warm-up sequence
            head_partitions: How many head groups to split the heads into,
                         group `p` pinned to core `p`. Must divide `heads`, and
                         must not exceed the cores in the target Vollo config.
        """
        super().__init__()

        self.mask = mask
        self.head_partitions = head_partitions

        if head_partitions is None:
            self.scan = vollo_torch.nn.Scan(
                _SlidingWindowAttentionStep(dim, heads, dim_head, bias, mask)
            )
        else:
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

            self.scans = nn.ModuleList(
                vollo_torch.nn.Scan(
                    _SlidingWindowAttentionStep(
                        dim,
                        heads // head_partitions,
                        dim_head,
                        bias,
                        mask,
                    )
                )
                for _ in range(head_partitions)
            )

        self.register_load_state_dict_pre_hook(self._load_any_partitioning)

        heads_per_scan = heads if head_partitions is None else heads // head_partitions

        # One initial window, shared by every group: they are all zeros, and a
        # buffer read in several partitions is still one compile-time constant.
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
            return self.scan(x, state)

        partials = []

        for p, scan in enumerate(self.scans):
            with vollo_torch.CorePartition([p]):
                partials.append(scan(x, state))

        return _tree_sum(partials)

    # Which projection weights split across the head groups, and along which
    # axis: the three inputs by output feature, `proj_o` by input feature.
    _SPLIT_AXES = (("proj_q", 0), ("proj_k", 0), ("proj_v", 0), ("proj_o", 1))

    def _load_any_partitioning(self, _module, state_dict, prefix, *_hook_args):
        """
        Let a checkpoint saved at one `head_partitions` load at any other.

        Partitioning changes where the projections run, not what they compute,
        so the two layouts differ by a concatenation: train once, then choose
        the partitioning per deployment.
        """
        step = f"{prefix}scan.step."
        scans = f"{prefix}scans."

        saved = sorted(
            {
                int(key[len(scans) :].split(".")[0])
                for key in state_dict
                if key.startswith(scans)
            }
        )

        if bool(saved) == (self.head_partitions is not None):
            return

        if saved:
            for name, axis in self._SPLIT_AXES:
                for suffix, dim in (("weight", axis), ("bias", 0)):
                    keys = [f"{scans}{p}.step.{name}.{suffix}" for p in saved]

                    if all(key in state_dict for key in keys):
                        state_dict[f"{step}{name}.{suffix}"] = torch.cat(
                            [state_dict.pop(key) for key in keys], dim=dim
                        )

            for key in [key for key in state_dict if key.startswith(scans)]:
                state_dict.pop(key)
        else:
            for name, axis in self._SPLIT_AXES:
                weight = state_dict.pop(f"{step}{name}.weight", None)

                if weight is not None:
                    for p, chunk in enumerate(weight.chunk(self.head_partitions, axis)):
                        state_dict[f"{scans}{p}.step.{name}.weight"] = chunk

                bias = state_dict.pop(f"{step}{name}.bias", None)

                if bias is not None:
                    for p, chunk in enumerate(bias.chunk(self.head_partitions)):
                        state_dict[f"{scans}{p}.step.{name}.bias"] = chunk


class _SlidingWindowAttentionStep(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        bias: bool,
        mask: bool,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.mask = mask

        inner_dim = heads * dim_head

        self.proj_q = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_k = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_v = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_o = nn.Linear(inner_dim, dim, bias=False)

        if mask:
            self.new_slot_bias = nn.Buffer(torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        """
        Args:
            x:     Tensor of shape (dim!), one timestep
            state: [K window, V window], both of shape (heads, window,
                   dim_head!), plus -- when masking warm-up -- the per-slot
                   additive score biases, of shape (window!)

        Returns:
            (Tensor of shape (dim!), the updated state)
        """
        k_win, v_win, bias_win = (
            state[0],
            state[1],
            state[2] if self.mask else None,
        )

        q = self.proj_q(x).view(self.heads, self.dim_head)  # [h dh!]
        k = self.proj_k(x).view(self.heads, self.dim_head)  # [h dh!]
        v = self.proj_v(x).view(self.heads, self.dim_head)  # [h dh!]

        # Slide the windows: evict the oldest entry, append this timestep's.
        k_win = torch.cat([k_win[:, 1:, :], k.unsqueeze(1)], dim=1)  # [h w dh!]
        v_win = torch.cat([v_win[:, 1:, :], v.unsqueeze(1)], dim=1)  # [h w dh!]
        if bias_win is not None:
            bias_win = torch.cat([bias_win[1:], self.new_slot_bias])  # [w!]

        # Both matmuls have two activations as operands, since the windows are
        # state rather than weights, so both are dynamic-weight matmuls (hence
        # `allow_dynamic_weights` in `_vm`). Each wants its matrix operand with
        # the contracted dimension second-innermost, and the scores contract over
        # features where the output contracts over timesteps -- hence the
        # transpose on one and not the other. It costs nothing: the compiler folds
        # it into how the window is read, and storing the K window the other way
        # round to begin with compiles to exactly the same program.
        #
        # The `1/sqrt(dim_head)` scale is applied here rather than on `q` at the
        # projection because it schedules better under `head_partitions`: the two
        # graphs are the same nodes in a different order, and order feeds the
        # compiler's op-to-core assignment (a wash unpartitioned, ~1% partitioned).
        #
        # [h 1 dh!] @ [h dh! w] -> [h 1 w!]
        scores = (q * self.scale).unsqueeze(1) @ k_win.transpose(1, 2)

        if self.mask:
            # Bias the scores
            scores = scores + bias_win  # [h 1 w!]

        attn = torch.softmax(scores, dim=-1)
        # [h 1 w!] @ [h w dh!] -> [h 1 dh!] -> [h dh!] -> [inner!]
        out = (attn @ v_win).squeeze(1).reshape(self.heads * self.dim_head)

        y = self.proj_o(out)

        new_state = [k_win, v_win]

        if self.mask:
            new_state.append(bias_win)

        return y, new_state


# Partitioning the heads pins each group's projections, window and attention to
# its own core, and turns `proj_o` into one partial projection per group that
# has to be summed. That sum is the only thing crossing cores, so it is summed
# pairwise rather than in a chain: the partials become available in parallel,
# and a chain would serialise them behind each other.
#
# It is a dial rather than a win, and how it pays depends on how much work lands
# on each core. `main`'s sweep uses six heads so that one group lands per core on
# the six-core configs and two per core on the three-core one; `_vm` picks the
# group count from the config for that reason.
#
# The number of heads a group holds is what matters: a group of one head is
# happy on one core, but at three heads per core the spaced latency goes up by
# 15-25% instead of 4-5%, so partition as finely as the config allows rather
# than into a few fat groups. `head_partitions=1` puts every head on core 0 and
# is the degenerate end of that, not a cheap "all cores" option.
def _tree_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    parts = list(partials)

    while len(parts) > 1:
        parts = [
            parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
            for i in range(0, len(parts), 2)
        ]

    return parts[0]


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

        The FFN uses an swiglu activation with expansion size of `expand`.

        Both sublayers are pre-norm and residual, and the norms and the residual
        adds live out here rather than inside the `Scan`: they are pointwise, so
        the streaming transform handles them wherever they sit, and keeping them
        out leaves `SlidingWindowAttention` a plain attention layer.

        Args:
            expand: FFN hidden dimension as a multiple of `dim`; every other
                    argument is `SlidingWindowAttention`'s, including
                    `head_partitions`, which partitions the attention and
                    leaves the FFN for the compiler to place
        """
        super().__init__()

        # SwiGLU multiplies the gate by the value, so the hidden dimension the
        # FFN comes back down from is `hidden`, not `2 * hidden`. Kept unfused
        # -- `ffn-swiglu.py` sweeps `fuse` and shows that folding `ffn_1` and
        # `ffn_2` into one wider projection can cost latency rather than save it.
        hidden = int(dim * expand)

        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attn = SlidingWindowAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            window_size=window_size,
            bias=bias,
            mask=mask,
            head_partitions=head_partitions,
        )

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
    dim_head: int = 64,
    heads: int = 6,
    partition: bool = False,
    expand: float = 2.0,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    input = torch.randn(2, dim)

    cores = CONFIGS[config].num_cores

    # Try to partition across all cores
    if partition and heads % cores == 0:
        head_partitions = cores
    else:
        head_partitions = None

    model = nn.Sequential().extend(
        SlidingWindowBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            window_size=window_size,
            mask=mask,
            head_partitions=head_partitions,
            expand=expand,
        )
        for _ in range(layers)
    )

    meta = dict(
        dim=dim,
        dim_head=dim_head,
        window=window_size,
        layers=layers,
        masked=mask,
    )

    if head_partitions is not None:
        meta["partitions"] = head_partitions

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        quick_compile=True,
        meta=meta,
    )


@beartype
def main(config: str = "V80") -> Generator:
    # Every size runs six heads, so that `head_partitions` can put one group on
    # each core of a six-core config and two on each core of a three-core one.
    for x in [
        dict(dim=32 * 6, dim_head=32, window_size=32, layers=1, mask=True),
        dict(dim=32 * 12, dim_head=32, window_size=32, layers=1, mask=True),
        dict(dim=32 * 12, dim_head=32, window_size=32, layers=6, mask=True),
    ]:
        for p in [True, False]:
            yield _vm(**x, config=config, partition=p)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
