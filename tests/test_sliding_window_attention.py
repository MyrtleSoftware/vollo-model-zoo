"""
The streamed sliding window against dense band-masked attention.

There is no external reference implementation to convert weights from here, but
the window plumbing -- what the scan state holds, which way round each window is
stored, how the warm-up slots are masked -- is exactly the part that is easy to
get subtly wrong, and it is all checkable against a dense attention written over
the whole sequence.
"""

import importlib

import torch
from beartype import beartype

sliding_window_attention = importlib.import_module(
    "vollo_model_zoo.models.sliding-window-attention"
)

SlidingWindowAttention = sliding_window_attention.SlidingWindowAttention

DIM, HEADS, DIM_HEAD, MLP_DIM = 32, 2, 16, 64


@beartype
def dense_reference(block, x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    The same block computed over the whole sequence at once, with a band mask
    doing what the rolling window does.

    Args:
        x: Tensor of shape (Time, dim)

    Returns:
        Tensor of shape (Time, dim)
    """
    step = block.step
    H, dh = step.heads, step.dim_head
    time = x.shape[0]

    h = step.attn_norm(x)
    q = step.to_q(h).view(time, H, dh) * step.scale
    k = step.to_k(h).view(time, H, dh)
    v = step.to_v(h).view(time, H, dh)

    scores = torch.einsum("ihd,jhd->hij", q, k)  # [heads, Time, Time]

    query = torch.arange(time).unsqueeze(-1)
    key = torch.arange(time).unsqueeze(0)
    attends_to = (key <= query) & (key > query - window_size)

    attn = torch.softmax(scores.masked_fill(~attends_to, float("-inf")), dim=-1)
    out = torch.einsum("hij,jhd->ihd", attn, v).reshape(time, H * dh)

    y = x + step.to_out(out)

    return y + step.mlp(step.mlp_norm(y))


@beartype
def build(window_size: int, mask_warmup: bool = True) -> SlidingWindowAttention:
    torch.manual_seed(0)

    return SlidingWindowAttention(
        dim=DIM,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_dim=MLP_DIM,
        window_size=window_size,
        mask_warmup=mask_warmup,
    ).eval()


@beartype
def test_matches_dense_band_masked_attention():
    """
    Every timestep, warm-up included: each query attends to itself and the
    `window_size - 1` timesteps before it, and to nothing else.
    """
    window_size = 4
    block = build(window_size)
    x = torch.randn(16, DIM)

    with torch.no_grad():
        got = block(x)
        expected = dense_reference(block, x, window_size)

    torch.testing.assert_close(got, expected)


@beartype
def test_window_at_least_the_sequence_length_is_full_causal_attention():
    """
    A window no shorter than the sequence never evicts anything, so the block
    degenerates to ordinary causal self-attention.
    """
    time = 8
    block = build(window_size=time)
    x = torch.randn(time, DIM)

    with torch.no_grad():
        got = block(x)
        expected = dense_reference(block, x, window_size=time)

    torch.testing.assert_close(got, expected)


@beartype
def test_unmasked_warmup_only_differs_while_the_window_fills():
    """
    `mask_warmup=False` is what the docstring says it is: the empty slots are
    attended to -- they hold zero keys, so they take a share of the softmax and
    contribute nothing -- for exactly as long as the window has empty slots, and
    from then on the block is unaffected.
    """
    window_size = 4
    block = build(window_size, mask_warmup=False)
    x = torch.randn(16, DIM)

    with torch.no_grad():
        got = block(x)
        expected = dense_reference(block, x, window_size)

    warm, streaming = got[: window_size - 1], got[window_size - 1 :]

    assert not torch.allclose(warm, expected[: window_size - 1])
    torch.testing.assert_close(streaming, expected[window_size - 1 :])
