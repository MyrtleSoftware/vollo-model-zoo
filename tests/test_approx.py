from functools import cache

import pytest
import torch
from beartype import beartype
from torch.utils.cpp_extension import load_inline
from vollo_torch import Fp32Activations

from vollo_model_zoo.vm import vollo_fn

_MANTISSA = 23
_EXP_SIGN = 9


@cache
def _round_impl(n: int, trunc: bool):
    source = """
      auto round_mantisa(at::Tensor x) -> void {{

        constexpr uint32_t o = 1;
        constexpr uint32_t n = {0};

        constexpr uint32_t trunc = {1};

        constexpr uint32_t _MANTISSA = {2};
        constexpr uint32_t _EXP_SIGN = {3};

        constexpr uint32_t mask = ((o << (n + _EXP_SIGN)) - o) << (_MANTISSA - n);
        constexpr uint32_t skip = trunc ? 0 : (o << (_MANTISSA - n - o));

        auto N = x.numel();
        auto * data = x.data_ptr<uint32_t>();

        for (int i = 0; i < N; i++) {{
            data[i] = (data[i] + skip) & mask;
        }}
      }}
    """.format(
        n,
        1 if trunc else 0,
        _MANTISSA,
        _EXP_SIGN,
    )

    cpp = load_inline(
        name="inline_extension",
        cpp_sources=source,
        functions="round_mantisa",
    )

    return cpp.round_mantisa  # type: ignore


def round_mantisa(x: torch.Tensor, n: int, trunc=False):
    """
    Take a tensor of float32 and round its mantissa to n bits.

    This runs on the CPU and will incur two copies if the input is on GPU.
    """
    assert x.dtype == torch.float32
    assert 0 <= n <= _MANTISSA

    if n == _MANTISSA:
        return x

    device = x.device

    x = x.cpu().clone(memory_format=torch.contiguous_format)

    assert x.is_contiguous()

    x = x.view(dtype=torch.uint32)
    _round_impl(n, trunc=trunc)(x)
    x = x.view(dtype=torch.float32)

    return x.to(device=device)


@pytest.fixture(scope="module")
@beartype
def all_bf16() -> torch.Tensor:
    """
    Generate a tensor containing all possible finite bfloat16 values.
    """
    # Generate all 16-bit bit patterns
    bits = torch.arange(-32768, 32767 + 1, dtype=torch.int16)
    exponent = (bits >> 7) & 0xFF
    finite_mask = exponent != 0xFF
    finite_bits = bits[finite_mask]
    bf16s = finite_bits.view(torch.bfloat16)
    return bf16s.sort()[0].to(torch.float32)


def exp_f32(x):
    with Fp32Activations():
        return torch.exp(x)


def test_exp(all_bf16):
    """
    Verify that Vollo's fp32 exp is at-least 24 bits of precision.
    """
    x = torch.cat([all_bf16, torch.linspace(-20, 2, steps=16_000)])
    x = x[x > -20]
    x = x[x < 2]

    y_ref = torch.exp(x)

    ref = {}

    for n in [24, 25, 26]:
        y_bn = round_mantisa(y_ref, n=n - 9)
        bn_max = (y_ref - y_bn).abs().max()
        bn_avg = (y_ref - y_bn).abs().sum()
        print(f"Ideal bf{n} error: {bn_max:.5e}, {bn_avg:.5e}")
        ref[n] = (bn_max, bn_avg)

    y_vol = vollo_fn(exp_f32, "V80")(x).to(torch.float32)

    vollo_max = (y_ref - y_vol).abs().max()
    vollo_avg = (y_ref - y_vol).abs().sum()
    print(f"Vollo fpNN error: {vollo_max:.5e}, {vollo_avg:.5e}")

    assert vollo_max < ref[24][0]
    assert vollo_avg < ref[24][1]


def recip_f32(x):
    """
    Given an f32 input, x, compute 1/x to almost full precision. This should be
    called outside of the Fp32Activation context.
    """
    # Now we want to compute 1 / z, this is a bf16 approx hence, ~6 bits of
    # mantissa precision
    y = 1 / x

    with Fp32Activations():
        # Newton-Raphson to compute reciprocal in fp32, converges quadratically
        # so 6 -> 12 -> 24 mantissa bits
        y = y * (2 - x * y)
        y = y * (2 - x * y)

    return y


def test_recip(all_bf16):
    x = torch.cat([all_bf16, torch.linspace(100, 100, steps=16_000)])
    x = x[x.abs() > 0.01]

    y_ref = 1 / x
    ref = {}

    for n in [31, 32]:
        y_bn = round_mantisa(y_ref, n=n - 9)
        bn_max = (y_ref - y_bn).abs().max()
        bn_avg = (y_ref - y_bn).abs().sum()
        print(f"Ideal bf{n} error: {bn_max:.5e}, {bn_avg:.5e}")
        ref[n] = (bn_max, bn_avg)

    y_vol = vollo_fn(recip_f32, "V80")(x).to(torch.float32)

    assert y_vol.isfinite().all()

    vollo_max = (y_ref - y_vol).abs().max()
    vollo_avg = (y_ref - y_vol).abs().sum()
    print(f"Vollo fpNN error: {vollo_max:.5e}, {vollo_avg:.5e}")

    assert vollo_max <= ref[31][0]
    assert vollo_avg <= ref[31][1]


def sigmoid_bf16_hi(x):
    """
    Compute sigmoid to approximately bf26 precision for bf16 inputs. This
    function _is_ compilable with Vollo. It should be called _outside_ of the
    Fp32Activations context.

    WARNING: for general fp32 inputs this is only bf18 precision.
    """
    # Prevent exp -> inf overflow
    x = -torch.clamp(x, min=-20)

    with Fp32Activations():
        # Precision of z is ~ bf25, this is effectively "full precision" for
        # bf16 inputs so the following reciprocal is exact. However, for fp32
        # inputs this has ~ 7 bits of error hence, when we do the Newton
        # iterations this compounds to ~14 bits of error which results in only
        # 18 bits of precision.
        z = 1.0 + torch.exp(x)

    return recip_f32(z)


def test_sigmoid(all_bf16):
    # Don't accidentally modify
    x = all_bf16.clone()

    y_ref_f32 = torch.sigmoid(x)

    assert y_ref_f32.isfinite().all()

    for n in [16, 18, 26]:
        y_ref_bn = round_mantisa(y_ref_f32, n=n - 9)
        ref_max = (y_ref_f32 - y_ref_bn).abs().max()
        ref_avg = (y_ref_f32 - y_ref_bn).abs().sum()
        print(f"Ideal bf{n} error: {ref_max:.5e}, {ref_avg:.5e}")

    # === bf16 sigmoid on vollo

    y_vol_b16 = vollo_fn(torch.sigmoid, "V80")(x).to(torch.float32)

    assert y_vol_b16.isfinite().all()

    vollo_max = (y_ref_f32 - y_vol_b16).abs().max()
    vollo_avg = (y_ref_f32 - y_vol_b16).abs().sum()
    print(f"Vollo bf16 error: {vollo_max:.5e}, {vollo_avg:.5e}")

    # === fp32 sigmoid

    y_vol_f32 = vollo_fn(sigmoid_bf16_hi, "V80")(x).to(torch.float32)

    assert y_vol_f32.isfinite().all()

    vollo_fp32_max = (y_ref_f32 - y_vol_f32).abs().max()
    vollo_fp32_avg = (y_ref_f32 - y_vol_f32).abs().sum()
    print(f"Vollo fp32 error: {vollo_fp32_max:.5e}, {vollo_fp32_avg:.5e}")

    # === Final comparison

    max_improvement = (vollo_max / vollo_fp32_max).log2()
    avg_improvement = (vollo_avg / vollo_fp32_avg).log2()
    print(f"Max improvement: {max_improvement:.2f} bits")
    print(f"Avg improvement: {avg_improvement:.2f} bits")

    assert avg_improvement > 9
    assert max_improvement > 10
