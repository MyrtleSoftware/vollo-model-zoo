"""
Checks AQ2-RT's step and readout against a reference implementation of the
paper's network. The reference is written here from the paper and Griffin, and
takes nothing from the model's code except its state dict. The comparison is in
fp64, with every parameter randomised.

Also checks that the compiled programs, run in Vollo's VM, match PyTorch.
"""

import numpy as np
import pytest
import torch
from beartype import beartype

from vollo_model_zoo.models import alphaqubit
from vollo_model_zoo.vm import CONFIGS, _vollo_compile

DIM, DIM_HEAD = 128, 32

F = torch.nn.functional


def rot(t: torch.Tensor) -> torch.Tensor:
    """Each head's channels `[x_a | x_b | y_a | y_b]` -> `[-x_b | x_a | -y_b | y_a]`."""
    x_a, x_b, y_a, y_b = t.unflatten(-1, (4, DIM_HEAD // 4)).unbind(-2)
    return torch.cat([-x_b, x_a, -y_b, y_a], dim=-1)


def rope_cos_sin(distance: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cos and sin of the RoPE angles, [N, 1, dh]. In each head, the first half of
    the channels uses the stabiliser's x and the second half its y, at
    frequencies 20 * 10^(-2i/8). Computed in fp32, as the model's table is,
    then converted to fp64.
    """
    coords = alphaqubit._stabiliser_coordinates(distance)
    freq = 20 * 10.0 ** (-2 * torch.arange(8) / 8)
    x, y = coords[:, :1] * freq, coords[:, 1:] * freq
    angle = torch.cat([x, x, y, y], dim=-1).unsqueeze(1)
    return angle.cos().double(), angle.sin().double()


def ref_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def ref_linear(sd, name, x):
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
    """Multi-head attention with heads of 32, torch.softmax and elementwise RoPE"""
    q, k, v = (
        ref_linear(sd, f"{p}.{name}", t).unflatten(-1, (-1, DIM_HEAD))  # [n h dh]
        for name, t in (("q", x), ("k", context), ("v", context))
    )
    if rope is not None:
        cos, sin = rope
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    scores = torch.einsum("nhd,mhd->hnm", q, k) / DIM_HEAD**0.5
    a = torch.einsum("hnm,mhd->nhd", torch.softmax(scores, dim=-1), v)
    return ref_linear(sd, f"{p}.proj_o", a.flatten(1))


def ref_layer(sd, p, x, context=None, rope=None):
    """The paper's pre-norm layer (Fig. S1a): attention, then a gated dense block"""
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


@beartype
def randomise(model: torch.nn.Module, dtype=torch.float64) -> torch.nn.Module:
    """Give every parameter, and the position table, a non-trivial value"""
    model.to(dtype)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if "norm" in name or name.endswith(".0.weight") and "dense" in name:
                t.copy_(torch.rand(t.shape, generator=gen) + 0.5)  # RMSNorm gains
            elif name.endswith(("bias", "decay", "observable", "position")):
                t.copy_(torch.randn(t.shape, generator=gen, dtype=t.dtype))
            elif name == "mean":  # the readout's 1 / N, recomputed in `dtype`
                t.fill_(1 / t.shape[-1])
    return model.eval()


def syndromes(distance: int, steps: int = 3, cycles: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    n = distance**2 - 1
    return torch.randint(0, 2, (steps, n, 2 * cycles), dtype=torch.float64)


@pytest.mark.parametrize("distance, layers", [(3, 2), (5, 2), (3, 3)])
@beartype
def test_step_matches_paper_reference(distance: int, layers: int):
    model = randomise(alphaqubit.AlphaQubit(distance, layers=layers))
    sd = model.state_dict()
    rope, x = rope_cos_sin(distance), syndromes(distance)

    state, expected = [torch.zeros(distance**2 - 1, DIM, dtype=torch.float64)] * 2, []
    for t in x:
        y, state = ref_step(sd, t, state, rope, layers)
        expected.append(y)
    torch.testing.assert_close(model(x), torch.stack(expected), atol=1e-10, rtol=1e-10)


@beartype
def test_readout_matches_paper_reference():
    model = randomise(alphaqubit.AlphaQubitReadout(5))
    x = torch.randn(24, DIM, dtype=torch.float64)
    expected = ref_readout(model.state_dict(), x, rope_cos_sin(5))
    torch.testing.assert_close(model(x), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "config, size",
    [
        ("V80", dict(part="step", distance=3)),
        ("V80", dict(part="step", distance=5)),
        ("IA-420f", dict(part="step", distance=3)),
        ("IA-840f", dict(part="step", distance=3)),
        ("V80", dict(part="readout", distance=5)),
    ],
    ids=lambda v: v if isinstance(v, str) else f"{v['part']}-d{v['distance']}",
)
@beartype
def test_vm_matches_pytorch(config: str, size: dict):
    """
    Compiles the program that `main()` compiles for the config and runs it in
    the VM: the step is streamed one group of cycles per inference, and the
    readout runs once. With `bit_accurate=False` the VM computes in fp32, so
    this tests the compilation (the streaming of both RG-LRU states,
    attention's matmuls) and not bf16 accuracy.
    """
    torch.manual_seed(0)
    args = alphaqubit.instance(**size, config=config)
    model = randomise(args["model"], dtype=torch.float32)
    if size["part"] == "step":
        x = syndromes(size["distance"], steps=4).float()
    else:
        x = torch.randn(size["distance"] ** 2 - 1, DIM)
    program = _vollo_compile(
        model,
        x,
        time_axis=args["time_axis"],
        config=CONFIGS[config],
        allow_dynamic_weights=args["allow_dynamic_weights"],
        quick_compile=args["quick_compile"],
    )
    vm = program.to_vm(bit_accurate=False)
    if size["part"] == "step":
        out = vm.run_timesteps(x.numpy(), 0, 0)
    else:
        out = vm.run(x.numpy())

    with torch.no_grad():
        expected = model(x).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_stabiliser_coordinates():
    for d in (3, 5, 7):
        coords = alphaqubit._stabiliser_coordinates(d)
        assert coords.shape == (d**2 - 1, 2)
        assert len({tuple(c) for c in coords.tolist()}) == d**2 - 1
        assert coords.min() == -0.5 and coords.max() == 0.5
        # (d - 1)^2 weight-4 plaquettes inside; the other 2 (d - 1) stabilisers
        # are weight-2, on the edges
        inside = ((coords > -0.5) & (coords < 0.5)).all(-1)
        assert inside.sum() == (d - 1) ** 2
