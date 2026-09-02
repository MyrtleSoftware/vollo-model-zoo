from collections.abc import Generator
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn

# The K/V window starts empty and fills as timesteps arrive, and a query must not
# attend to the slots that have not been filled yet. Rather than detect emptiness
# at runtime, the key gets one extra feature carrying an additive score bias: the
# query's matching feature is a constant 1, a real key appends a 0 (no bias), and
# the window's initial state holds `_EMPTY_BIAS` in that feature. The bias then
# rides through the score matmul that is happening anyway, and the softmax gives
# an empty slot a weight of exp(-inf) == 0. Vollo has propagated infinities
# through `exp` correctly since SDK 28.0.0; on an older compiler a large negative
# value underflows to the same weight.
_EMPTY_BIAS = float("-inf")


class SlidingWindowAttention(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        window_size: int,
        bias: bool = True,
        mask_warmup: bool = True,
    ):
        """
        A pre-norm self-attention sublayer that is causal and windowed: each
        timestep attends to itself and the `window_size - 1` timesteps before it.

        The rolling K/V window is held as `vollo_torch.nn.Scan` state, so the
        block streams: one timestep in, one timestep out, with a fixed amount of
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
                         of a sequence. Costs one extra key feature (see
                         `_EMPTY_BIAS`); turn it off if the accelerator is only
                         ever read after streaming in a warm-up sequence
        """
        super().__init__()

        self.step = _SlidingWindowAttentionStep(dim, heads, dim_head, bias, mask_warmup)
        self.scan = vollo_torch.nn.Scan(self.step)

        k_0 = torch.zeros(heads, window_size, dim_head + int(mask_warmup))

        if mask_warmup:
            k_0[:, :, -1] = _EMPTY_BIAS

        self.k_0 = nn.Buffer(k_0, persistent=False)

        self.v_0 = nn.Buffer(
            torch.zeros(heads, window_size, dim_head), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Time, dim)

        Returns:
            x: Tensor of shape (Time, dim)
        """
        return self.scan(x, [self.k_0, self.v_0])


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

        self.attn_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

        if mask_warmup:
            # The mask feature: the query weights the bias by one, and an
            # arriving key is real so it carries no bias. Buffers rather than
            # `new_ones`/`new_zeros`, so they are compile-time constants with no
            # data dimension.
            self.q_mask_feature = nn.Buffer(torch.ones(heads, 1), persistent=False)
            self.k_mask_feature = nn.Buffer(torch.zeros(heads, 1), persistent=False)

    def forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        """
        Args:
            x:     Tensor of shape (dim!), one timestep
            state: [K window, V window] of shapes (heads, window, f!) and
                   (heads, window, dim_head!), where f is dim_head plus the mask
                   feature

        Returns:
            (Tensor of shape (dim!), the updated state)
        """
        k_win, v_win = state
        H, dh = self.heads, self.dim_head

        h = self.attn_norm(x)
        # The scale goes on the query rather than the scores so that the mask
        # bias, which is applied inside the score matmul, is not scaled with them.
        q = self.to_q(h).view(H, dh) * self.scale  # [h dh!]
        k = self.to_k(h).view(H, dh)  # [h dh!]
        v = self.to_v(h).view(H, dh)  # [h dh!]

        if self.mask_warmup:
            q = torch.cat([q, self.q_mask_feature], dim=-1)  # [h f!]
            k = torch.cat([k, self.k_mask_feature], dim=-1)  # [h f!]

        # Slide the windows: evict the oldest entry, append this timestep's.
        k_win = torch.cat([k_win[:, 1:, :], k.unsqueeze(1)], dim=1)  # [h w f!]
        v_win = torch.cat([v_win[:, 1:, :], v.unsqueeze(1)], dim=1)  # [h w dh!]

        # Both matmuls have two activations as operands, since the windows are
        # state rather than weights, so both are dynamic-weight matmuls (hence
        # `allow_dynamic_weights` in `_vm`). Each wants its matrix operand with
        # the contracted dimension second-innermost, and the scores contract over
        # features where the output contracts over timesteps -- hence the
        # transpose on one and not the other. It costs nothing: the compiler folds
        # it into how the window is read, and storing the K window the other way
        # round to begin with compiles to exactly the same program.
        #
        # [h 1 f!] @ [h f! w] -> [h 1 w!]
        attn = torch.softmax(q.unsqueeze(1) @ k_win.transpose(1, 2), dim=-1)
        # [h 1 w!] @ [h w dh!] -> [h 1 dh!] -> [h dh!] -> [inner!]
        out = (attn @ v_win).squeeze(1).reshape(H * dh)

        x = x + self.to_out(out)

        return x, [k_win, v_win]


@beartype
def _vm(
    dim: int,
    heads: int,
    dim_head: int,
    window_size: int,
    layers: int,
    config: str,
    mask_warmup: bool = True,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(2, dim)

    model = nn.Sequential().extend(
        SlidingWindowAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            window_size=window_size,
            mask_warmup=mask_warmup,
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
        dict(dim=288, heads=3, dim_head=96, window_size=16, layers=2),
        # ~1M parameter baseline: with heads * dim_head == dim, a sublayer holds
        # its four projections and so 4 * dim^2 + O(dim) parameters, and
        # 3 * 4 * 288^2 approx 1M
        dict(dim=288, heads=3, dim_head=96, window_size=16, layers=3),
        dict(dim=384, heads=6, dim_head=64, window_size=32, layers=2),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
