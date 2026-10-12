"""
The folded RoPE projections of `_Attention` against explicit RoPE.

The model never rotates q and k: it adds `W_rot x * sin`, where `W_rot` is the
projection's weight with its output channels rotated. That equals rotating the
projected q and k (`t * cos + rot(t) * sin`) when `W_rot` is tied to `W` this way.
"""

import importlib

import torch

alphaqubit = importlib.import_module("vollo_model_zoo.models.alphaqubit")

DISTANCE, DIM, HEADS, DIM_HEAD = 3, 64, 2, 32


def rot(t: torch.Tensor) -> torch.Tensor:
    """Each head's channels `[x_a | x_b | y_a | y_b]` -> `[-x_b | x_a | -y_b | y_a]`."""
    *lead, width = t.shape
    x_a, x_b, y_a, y_b = t.view(*lead, -1, 4, DIM_HEAD // 4).unbind(-2)
    return torch.stack([-x_b, x_a, -y_b, y_a], dim=-2).reshape(*lead, width)


def explicit_rope_attention(attn, x: torch.Tensor, cos, sin) -> torch.Tensor:
    q_proj, k_proj, v_proj, *_ = attn.proj[0]

    def heads(t):
        return t.view(-1, HEADS, DIM_HEAD).transpose(0, 1)  # [h N dh]

    q, k, v = heads(q_proj(x)), heads(k_proj(x)), heads(v_proj(x))
    q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    scores = torch.softmax(q @ k.transpose(1, 2) * DIM_HEAD**-0.5, dim=-1)
    return attn.proj_o((scores @ v).transpose(0, 1).reshape(-1, HEADS * DIM_HEAD))


def test_folded_rope_matches_explicit_rope():
    torch.manual_seed(0)
    cos, sin = alphaqubit._rope_tables(DISTANCE, DIM_HEAD)
    attn = alphaqubit._Attention(DIM, HEADS, DIM_HEAD, 1, rope=(cos, sin))
    q, k, _, q_rot, k_rot = attn.proj[0]
    with torch.no_grad():
        for plain, folded in ((q, q_rot), (k, k_rot)):
            # rot acts on output channels: apply it to the weight's columns
            folded.weight.copy_(rot(plain.weight.T).T)
            folded.bias.copy_(rot(plain.bias))

    x = torch.randn(DISTANCE**2 - 1, DIM)
    torch.testing.assert_close(
        attn(x), explicit_rope_attention(attn, x, cos, sin), atol=1e-5, rtol=1e-5
    )
