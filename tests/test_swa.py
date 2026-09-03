"""
The streamed sliding window against dense band-masked attention.

There is no external reference implementation to convert weights from here, but
the window plumbing -- what the scan state holds, which way round each window is
stored, how the warm-up slots are masked -- is exactly the part that is easy to
get subtly wrong, and it is all checkable against a dense attention written over
the whole sequence.
"""

import pytest
import torch
from beartype import beartype

from vollo_model_zoo.models.swa import SlidingWindowAttention, SlidingWindowBlock

DIM, HEADS, DIM_HEAD = 32, 2, 16


@beartype
def dense_reference(
    layer: SlidingWindowAttention, x: torch.Tensor, window_size: int
) -> torch.Tensor:
    """
    The same attention computed over the whole sequence at once, with a band
    mask doing what the rolling window does.

    Args:
        x: Tensor of shape (Time, dim)

    Returns:
        Tensor of shape (Time, dim)
    """
    step = layer.scan.step
    H, dh = step.heads, step.dim_head
    time = x.shape[0]

    q = step.proj_q(x).view(time, H, dh) * step.scale
    k = step.proj_k(x).view(time, H, dh)
    v = step.proj_v(x).view(time, H, dh)

    scores = torch.einsum("ihd,jhd->hij", q, k)  # [heads, Time, Time]

    query = torch.arange(time).unsqueeze(-1)
    key = torch.arange(time).unsqueeze(0)
    attends_to = (key <= query) & (key > query - window_size)

    attn = torch.softmax(scores.masked_fill(~attends_to, float("-inf")), dim=-1)
    out = torch.einsum("hij,jhd->ihd", attn, v).reshape(time, H * dh)

    # The output projection lives on the layer, not the scan step, so that
    # `head_partitions` does not split it.
    return layer.proj_o(out)


@beartype
def build_layer(
    window_size: int,
    mask: bool = True,
    heads: int = HEADS,
    dim_head: int = DIM_HEAD,
    **kwargs,
) -> SlidingWindowAttention:
    torch.manual_seed(0)

    return SlidingWindowAttention(
        dim=DIM,
        heads=heads,
        dim_head=dim_head,
        window_size=window_size,
        mask=mask,
        **kwargs,
    ).eval()


@beartype
def test_matches_dense_band_masked_attention():
    """
    Every timestep, warm-up included: each query attends to itself and the
    `window_size - 1` timesteps before it, and to nothing else.
    """
    window_size = 4
    layer = build_layer(window_size)
    x = torch.randn(16, DIM)

    with torch.no_grad():
        got = layer(x)
        expected = dense_reference(layer, x, window_size)

    torch.testing.assert_close(got, expected)


@beartype
def test_window_at_least_the_sequence_length_is_full_causal_attention():
    """
    A window no shorter than the sequence never evicts anything, so the layer
    degenerates to ordinary causal self-attention.
    """
    time = 8
    layer = build_layer(window_size=time)
    x = torch.randn(time, DIM)

    with torch.no_grad():
        got = layer(x)
        expected = dense_reference(layer, x, window_size=time)

    torch.testing.assert_close(got, expected)


@beartype
def test_unmasked_warmup_only_differs_while_the_window_fills():
    """
    `mask=False` is what the docstring says it is: the empty slots are
    attended to -- they hold zero keys, so they take a share of the softmax and
    contribute nothing -- for exactly as long as the window has empty slots, and
    from then on the layer is unaffected.
    """
    window_size = 4
    layer = build_layer(window_size, mask=False)
    x = torch.randn(16, DIM)

    with torch.no_grad():
        got = layer(x)
        expected = dense_reference(layer, x, window_size)

    warm, streaming = got[: window_size - 1], got[window_size - 1 :]

    assert not torch.allclose(warm, expected[: window_size - 1])
    torch.testing.assert_close(streaming, expected[window_size - 1 :])


@beartype
def test_block_is_two_pre_norm_residual_sublayers():
    """
    `SlidingWindowBlock` wires the sublayers the way its docstring draws them:
    the attention reads the normed input and not the raw one, the FFN reads the
    normed attention output, and each sublayer adds back the tensor that went
    into its norm.
    """
    window_size = 4
    torch.manual_seed(0)
    block = SlidingWindowBlock(
        dim=DIM, heads=HEADS, dim_head=DIM_HEAD, window_size=window_size
    ).eval()
    x = torch.randn(16, DIM)

    with torch.no_grad():
        got = block(x)

        attended = x + dense_reference(block.attn, block.attn_norm(x), window_size)
        normed = block.ffn_norm(attended)
        gated = torch.nn.functional.silu(block.ffn_1(normed)) * block.ffn_2(normed)
        expected = attended + block.ffn_3(gated)

    torch.testing.assert_close(got, expected)


@beartype
@pytest.mark.parametrize("head_partitions", [1, 2, 3, 6])
def test_head_partitioning_computes_the_same_attention(head_partitions: int):
    """
    Partitioning moves where the projections run, not what they compute, so a
    partitioned layer matches an unpartitioned one once its checkpoint is
    loaded -- the load hook doing the splitting -- to within the reassociation
    of `proj_o`'s sum over the head groups.
    """
    window_size = 4
    plain = build_layer(window_size, heads=6, dim_head=DIM // 6)
    partitioned = build_layer(
        window_size, heads=6, dim_head=DIM // 6, head_partitions=head_partitions
    )
    partitioned.load_state_dict(plain.state_dict())

    x = torch.randn(16, DIM)

    with torch.no_grad():
        torch.testing.assert_close(partitioned(x), plain(x))


@beartype
@pytest.mark.parametrize("head_partitions", [1, 2, 3, 6])
def test_a_partitioned_checkpoint_loads_unpartitioned(head_partitions: int):
    """
    And back the other way, so the partitioning is a deployment choice rather
    than something baked into the weights.
    """
    window_size = 4
    partitioned = build_layer(
        window_size, heads=6, dim_head=DIM // 6, head_partitions=head_partitions
    )
    plain = build_layer(window_size, heads=6, dim_head=DIM // 6)
    plain.load_state_dict(partitioned.state_dict())

    x = torch.randn(16, DIM)

    with torch.no_grad():
        torch.testing.assert_close(plain(x), partitioned(x))


@beartype
@pytest.mark.parametrize("saved,loaded", [(2, 3), (3, 6), (6, 2), (1, 6), (6, 1)])
def test_a_partitioned_checkpoint_loads_at_another_partitioning(
    saved: int, loaded: int
):
    """
    And between two partitionings, which is the pair `_vm` actually asks for:
    it partitions over the cores of the target config, so a checkpoint trained
    against a six-core V80 has to load into the three-way split of an IA-840f.
    """
    window_size = 4
    src = build_layer(window_size, heads=6, dim_head=DIM // 6, head_partitions=saved)
    dst = build_layer(window_size, heads=6, dim_head=DIM // 6, head_partitions=loaded)
    dst.load_state_dict(src.state_dict())

    x = torch.randn(16, DIM)

    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x))


@beartype
def test_partitioning_rejects_uneven_head_groups():
    """
    Uneven head groups leave one core holding more heads than the rest, and
    that core sets the latency, so it is an error rather than a warning. How
    many cores the config actually has is not visible here, so asking for more
    groups than cores is left to the compiler.
    """
    with pytest.raises(ValueError, match="multiple of head_partitions"):
        build_layer(4, heads=6, dim_head=DIM // 6, head_partitions=4)

    with pytest.raises(ValueError, match="between 1 and the number of heads"):
        build_layer(4, heads=6, dim_head=DIM // 6, head_partitions=7)
