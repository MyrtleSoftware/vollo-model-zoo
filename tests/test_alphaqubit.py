"""
`_Attention` against attention with explicit RoPE, and the step and the readout
against the paper's network written out from the paper (fp64, every parameter
randomised).
"""

import importlib

import pytest
import torch

alphaqubit = importlib.import_module("vollo_model_zoo.models.alphaqubit")

DISTANCE, DIM, HEADS, DIM_HEAD = 3, 64, 2, 32
DIM_STEP = 128

F = torch.nn.functional


def rot(t: torch.Tensor) -> torch.Tensor:
    """Each head's channels `[x_a | x_b | y_a | y_b]` -> `[-x_b | x_a | -y_b | y_a]`."""
    *lead, width = t.shape
    x_a, x_b, y_a, y_b = t.view(*lead, -1, 4, DIM_HEAD // 4).unbind(-2)
    return torch.stack([-x_b, x_a, -y_b, y_a], dim=-2).reshape(*lead, width)


def explicit_rope_attention(attn, x: torch.Tensor, cos, sin) -> torch.Tensor:
    def heads(t):
        return t.view(-1, HEADS, DIM_HEAD).transpose(0, 1)  # [h N dh]

    q, k, v = (heads(p[0](x)) for p in (attn.q, attn.k, attn.v))
    q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    scores = torch.softmax(q @ k.transpose(1, 2) * DIM_HEAD**-0.5, dim=-1)
    return attn.proj_o((scores @ v).transpose(0, 1).reshape(-1, HEADS * DIM_HEAD))


def test_rope_matches_explicit_rope():
    torch.manual_seed(0)
    cos, sin = alphaqubit._rope_tables(DISTANCE, DIM_HEAD)
    attn = alphaqubit._Attention(DIM, HEADS, DIM_HEAD, 1, rope=(cos, sin))

    x = torch.randn(DISTANCE**2 - 1, DIM)
    torch.testing.assert_close(
        attn(x), explicit_rope_attention(attn, x, cos, sin), atol=1e-5, rtol=1e-5
    )


# The paper's network, written out from the paper and Griffin, independently of
# the model's code: it reads only the state dict, joining the per-head-group
# q, k and v. Fp64, so any difference beyond rounding is a bug, whatever the
# model's spread over the cores.


def ref_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def ref_linear(sd, name, x):
    if f"{name}.weight" not in sd:  # one Linear per head group: join them
        groups = sum(k.startswith(f"{name}.") and k.endswith(".weight") for k in sd)
        weight = torch.cat([sd[f"{name}.{g}.weight"] for g in range(groups)])
        bias = torch.cat([sd[f"{name}.{g}.bias"] for g in range(groups)])
        return x @ weight.T + bias
    return x @ sd[f"{name}.weight"].T + sd[f"{name}.bias"]


def ref_rglru(sd, p, x, state):
    """Griffin's recurrent block, as in the paper's Fig. S3"""
    h = ref_norm(x, sd[f"{p}.norm.weight"])
    u = ref_linear(sd, f"{p}.inp", h)
    r = torch.sigmoid(ref_linear(sd, f"{p}.proj_r", u))
    i = torch.sigmoid(ref_linear(sd, f"{p}.proj_i", u))
    a = torch.exp(-8 * F.softplus(sd[f"{p}.decay"]) * r)
    state = a * state + torch.sqrt(1 - a**2) * (i * u)
    gated = state * F.gelu(ref_linear(sd, f"{p}.gate", h))
    return x + ref_linear(sd, f"{p}.out", gated), state


def ref_attention(sd, p, x, context, rope):
    """Multi-head attention, heads of 32, torch.softmax, explicit spatial RoPE"""
    q, k, v = (
        ref_linear(sd, f"{p}.{name}", t).unflatten(-1, (-1, DIM_HEAD))  # [n h dh]
        for name, t in (("q", x), ("k", context), ("v", context))
    )
    if rope is not None:
        cos, sin = (t.unsqueeze(1) for t in rope)  # [n 1 dh]
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    scores = torch.einsum("nhd,mhd->hnm", q, k) / DIM_HEAD**0.5
    a = torch.einsum("hnm,mhd->nhd", torch.softmax(scores, dim=-1), v)
    return ref_linear(sd, f"{p}.proj_o", a.flatten(1))


def ref_layer(sd, p, x, context=None, rope=None):
    """The paper's serial pre-norm layer (Fig. S1a): attention, then gated dense"""
    h = ref_norm(x, sd[f"{p}.norm_attn.weight"])
    c = h if context is None else ref_norm(context, sd[f"{p}.norm_attn.weight"])
    x = x + ref_attention(sd, f"{p}.attn", h, c, rope)
    h = ref_norm(x, sd[f"{p}.norm_dense.weight"])
    up, gate = (ref_linear(sd, f"{p}.{name}", h) for name in ("up", "gate"))
    return x + ref_linear(sd, f"{p}.down", F.gelu(up) * gate)


def ref_step(sd, syndromes, state, rope, layers):
    p = "scan.step"
    x = ref_linear(sd, f"{p}.embed", syndromes) + sd[f"{p}.position"]
    x, rnn_in = ref_rglru(sd, f"{p}.rnn_in", x, state[0])
    for layer in range(layers):
        x = ref_layer(sd, f"{p}.layers.{layer}", x, rope=rope)
    x, rnn_out = ref_rglru(sd, f"{p}.rnn_out", x, state[1])
    return x, [rnn_in, rnn_out]


def ref_readout(sd, x, rope):
    for layer in range(2):
        x = ref_layer(sd, f"mix.{layer}", x, rope=rope)
    query = x.mean(0, keepdim=True) + sd["observable"]
    for layer in range(2):
        query = ref_layer(sd, f"cross.{layer}", query, context=x)
    for block in range(2):
        h = ref_norm(query, sd[f"dense.{block}.0.weight"])
        hidden = F.gelu(ref_linear(sd, f"dense.{block}.1", h))
        query = query + ref_linear(sd, f"dense.{block}.3", hidden)
    return ref_linear(sd, "logit", query).squeeze(0)


def randomise_all(model: torch.nn.Module) -> torch.nn.Module:
    """Every parameter and the position table non-trivial, in fp64"""
    model.double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if "norm" in name or name.endswith(".0.weight") and "dense" in name:
                t.copy_(torch.rand(t.shape, generator=gen) + 0.5)  # RMSNorm gains
            elif name.endswith(("bias", "decay", "observable", "position")):
                t.copy_(torch.randn(t.shape, generator=gen, dtype=t.dtype))
            elif name == "mean":  # the readout's 1 / N, built in fp32
                t.fill_(1 / t.shape[-1])
    return model


@pytest.mark.parametrize("distance, layers", [(3, 2), (5, 2), (3, 3)])
@pytest.mark.parametrize("head_partitions", [1, 4])
def test_step_matches_paper_reference(distance, layers, head_partitions):
    torch.manual_seed(0)
    n = distance**2 - 1
    args = distance, 6, DIM_STEP, layers, 4, 4, head_partitions
    model = randomise_all(alphaqubit.AlphaQubit(*args)).eval()
    sd = model.state_dict()
    rope = tuple(t.double() for t in alphaqubit._rope_tables(distance, DIM_HEAD))
    syndromes = torch.randint(0, 2, (3, n, 12)).double()

    state, expected = [torch.zeros(n, DIM_STEP, dtype=torch.float64)] * 2, []
    for t in syndromes:
        x, state = ref_step(sd, t, state, rope, layers)
        expected.append(x)
    torch.testing.assert_close(
        model(syndromes), torch.stack(expected), atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("head_partitions", [1, 4])
def test_readout_matches_paper_reference(head_partitions):
    torch.manual_seed(0)
    model = alphaqubit.AlphaQubitReadout(5, DIM_STEP, 4, 4, head_partitions)
    model = randomise_all(model).eval()
    rope = tuple(t.double() for t in alphaqubit._rope_tables(5, DIM_HEAD))
    x = torch.randn(24, DIM_STEP, dtype=torch.float64)
    torch.testing.assert_close(
        model(x), ref_readout(model.state_dict(), x, rope), atol=1e-10, rtol=1e-10
    )
