from collections.abc import Generator
from pathlib import Path

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
        mask_warmup: bool = True,
    ):
        """
        A self-attention sublayer that is causal and windowed: each timestep
        attends to itself and the `window_size - 1` timesteps before it.

        The rolling K/V window is held as `vollo_torch.nn.Scan` state, so it
        streams: one timestep in, one timestep out, with a fixed amount of
        state and a fixed amount of work per timestep however long the sequence
        gets. That is what makes attention viable at low latency -- full
        self-attention keeps the whole sequence resident and its per-timestep
        cost grows with the sequence length.

        Args:
            dim:         Input/output dimension
            heads:       Number of attention heads
            dim_head:    Dimension of each attention head
            window_size: Number of timesteps a query attends over, itself
                         included; also the length of the K/V scan state
            bias:        Whether the linear layers use biases
            mask_warmup: Whether to mask the window slots that have not been
                         filled yet, over the first `window_size - 1` timesteps
                         of a sequence. Costs a third scan state and a pointwise
                         add; turn it off if the accelerator is only ever read
                         after streaming in a warm-up sequence
        """
        super().__init__()

        self.mask_warmup = mask_warmup

        self.scan = vollo_torch.nn.Scan(
            _SlidingWindowAttentionStep(dim, heads, dim_head, bias, mask_warmup)
        )

        self.k_0 = nn.Buffer(
            torch.zeros(heads, window_size, dim_head), persistent=False
        )

        self.v_0 = nn.Buffer(
            torch.zeros(heads, window_size, dim_head), persistent=False
        )

        if self.mask_warmup:
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

        if self.mask_warmup:
            state.append(self.bias_0)

        return self.scan(x, state)


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
        mask_warmup: bool = True,
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
                    argument is `SlidingWindowAttention`'s
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
            mask_warmup=mask_warmup,
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


class _SlidingWindowAttentionStep(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        bias: bool,
        mask_warmup: bool,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.mask_warmup = mask_warmup

        inner_dim = heads * dim_head

        self.proj_q = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_k = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_v = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_o = nn.Linear(inner_dim, dim, bias=bias)

        if mask_warmup:
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
            state[2] if self.mask_warmup else None,
        )

        q = self.proj_q(x).view(self.heads, self.dim_head) * self.scale  # [h dh!]
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
        # [h 1 dh!] @ [h dh! w] -> [h 1 w!]
        scores = q.unsqueeze(1) @ k_win.transpose(1, 2)

        if self.mask_warmup:
            # Bias the scores
            scores = scores + bias_win  # [h 1 w!]

        attn = torch.softmax(scores, dim=-1)
        # [h 1 w!] @ [h w dh!] -> [h 1 dh!] -> [h dh!] -> [inner!]
        out = (attn @ v_win).squeeze(1).reshape(self.heads * self.dim_head)

        y = self.proj_o(out)

        new_state = [k_win, v_win]

        if self.mask_warmup:
            new_state.append(bias_win)

        return y, new_state


@beartype
def _vm(
    dim: int,
    heads: int,
    dim_head: int,
    window_size: int,
    layers: int,
    config: str,
    mask_warmup: bool = True,
    expand: float = 2.0,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(2, dim)

    model = nn.Sequential().extend(
        SlidingWindowBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            window_size=window_size,
            mask_warmup=mask_warmup,
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
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            ffn=int(dim * expand),
            window=window_size,
            layers=layers,
            masked=mask_warmup,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=192, heads=3, dim_head=64, window_size=16, layers=1),
        dict(dim=192, heads=3, dim_head=64, window_size=64, layers=1),
        dict(dim=288, heads=3, dim_head=96, window_size=16, layers=1),
        dict(dim=288, heads=3, dim_head=96, window_size=64, layers=1),
        dict(
            dim=288,
            heads=3,
            dim_head=96,
            window_size=64,
            layers=1,
            mask_warmup=False,
        ),
        # ~1M parameter baseline: with heads * dim_head == dim and expand == 2,
        # a block holds 4 * dim^2 in the attention projections and 6 * dim^2 in
        # the FFN, so 10 * dim^2 + O(dim), and 10 * 320^2 approx 1M
        dict(dim=320, heads=5, dim_head=64, window_size=16, layers=1),
        dict(dim=288, heads=3, dim_head=96, window_size=16, layers=2),
        dict(dim=384, heads=6, dim_head=64, window_size=32, layers=2),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
