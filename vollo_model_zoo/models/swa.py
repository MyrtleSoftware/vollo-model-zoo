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
                        dim, heads // head_partitions, dim_head, bias, mask
                    )
                )
                for _ in range(head_partitions)
            )

        self.proj_o = nn.Linear(heads * dim_head, dim, bias=bias)

        # Same zeros initial buffer for each head partition

        heads_per_scan = heads if head_partitions is None else heads // head_partitions

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

    def _load_any_partitioning(self, _module, state_dict, prefix, *_hook_args):
        """
        Let a checkpoint saved at one `head_partitions` load at any other.
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

        # The projections that split across the head groups, by output feature.
        # `proj_o` is not among them: it lives outside the partitioning.
        split_axes = ("proj_q", "proj_k", "proj_v")

        if saved:
            for name in split_axes:
                for suffix in ("weight", "bias"):
                    keys = [f"{scans}{p}.step.{name}.{suffix}" for p in saved]

                    if all(key in state_dict for key in keys):
                        state_dict[f"{step}{name}.{suffix}"] = torch.cat(
                            [state_dict.pop(key) for key in keys]
                        )

            for key in [key for key in state_dict if key.startswith(scans)]:
                state_dict.pop(key)
        else:
            for name in split_axes:
                for suffix in ("weight", "bias"):
                    tensor = state_dict.pop(f"{step}{name}.{suffix}", None)

                    if tensor is None:
                        continue

                    for p, chunk in enumerate(tensor.chunk(self.head_partitions)):
                        state_dict[f"{scans}{p}.step.{name}.{suffix}"] = chunk


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
        """
        The scan function, see SlidingWindowAttention for arguments.
        """
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.mask = mask

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

        # [h 1 dh!] @ [h dh! w] -> [h 1 w!]
        scores = (q * self.scale).unsqueeze(1) @ k_win.transpose(1, 2)

        if self.mask:
            # Bias the scores
            scores = scores + bias_win  # [h 1 w!]

        attn = torch.softmax(scores, dim=-1)

        # [h 1 w!] @ [h w dh!] -> [h 1 dh!] -> [h dh!] -> [inner!]
        out = (attn @ v_win).squeeze(1).reshape(self.heads * self.dim_head)

        new_state = [k_win, v_win]

        if self.mask:
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
        # ~1M parameter baseline, the size models are compared at: with six
        # 32-wide heads and expand == 2, a block holds 6 * dim^2 in the FFN and
        # 768 * dim + O(dim) in the attention projections, and that reaches 1M
        # at dim = 352.
        dict(dim=32 * 11, dim_head=32, window_size=32, layers=1, mask=True),
        dict(dim=32 * 12, dim_head=32, window_size=32, layers=1, mask=True),
        dict(dim=32 * 12, dim_head=32, window_size=32, layers=6, mask=True),
    ]:
        for p in [True, False]:
            yield _vm(**x, config=config, partition=p)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
