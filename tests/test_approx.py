import torch
from beartype import beartype
from vollo_torch import Fp32Activations

from vollo_model_zoo.approx import round_mantisa, sigmoid
from vollo_model_zoo.vm import vollo_fn


@beartype
def gen_all_bf16() -> torch.Tensor:
    """
    Generate a tensor containing all possible finite bfloat16 values.
    """
    # Generate all 16-bit bit patterns
    bits = torch.arange(-32768, 32767 + 1, dtype=torch.int16)
    exponent = (bits >> 7) & 0xFF
    finite_mask = exponent != 0xFF
    finite_bits = bits[finite_mask]
    bf16s = finite_bits.view(torch.bfloat16)
    return bf16s.sort()[0]


def recip_f32(x):
    """
    Given an f32 x, compute 1/x to full precision. This should
    be called outside of the Fp32Activation context.
    """
    # Now we want to compute 1 / z, this is a bf16 approx
    # hence, ~6 bits of precision
    y = 1 / x

    with Fp32Activations():
        # Newton-Raphson to compute reciprocal in fp32,
        # converges quadratically so 6 -> 12 -> 24 bits
        y = y * (2 - x * y)
        y = y * (2 - x * y)

    return y


def sigmoid_f32(x):
    """
    Compute sigmoid to a similar precision as Vollo's exp32
    """
    # Prevent exp -> inf overflow
    x = torch.clamp(x, min=-20)
    x = -x

    with Fp32Activations():
        z = 1.0 + torch.exp(x)

    return recip_f32(z)


def test_meta():
    x = gen_all_bf16().to(torch.float32)

    y_ref_f32 = torch.sigmoid(x)

    assert y_ref_f32.isfinite().all()

    for n in [16, 26]:
        y_ref_bn = round_mantisa(y_ref_f32, n=n - 9)
        ref_max = (y_ref_f32 - y_ref_bn).abs().max()
        ref_avg = (y_ref_f32 - y_ref_bn).abs().sum()
        print(f"Ideal bf{n} error: {ref_max:.5e}, {ref_avg:.5e}")

    # === The python table approximation

    y_vol_approx = sigmoid(x)

    assert y_vol_approx.isfinite().all()

    approx_max = (y_ref_f32 - y_vol_approx).abs().max()
    approx_avg = (y_ref_f32 - y_vol_approx).abs().sum()
    print(f"LuT'd bf16 error: {approx_max:.5e}, {approx_avg:.5e}")

    # === bf16 sigmoid

    y_vol_b16 = vollo_fn(torch.sigmoid, "V80")(x).to(torch.float32)

    assert y_vol_b16.isfinite().all()

    vollo_max = (y_ref_f32 - y_vol_b16).abs().max()
    vollo_avg = (y_ref_f32 - y_vol_b16).abs().sum()
    print(f"Vollo bf16 error: {vollo_max:.5e}, {vollo_avg:.5e}")

    # === fp32 sigmoid

    y_vol_f32 = vollo_fn(sigmoid_f32, "V80")(x).to(torch.float32)

    assert y_vol_f32.isfinite().all()

    vollo_fp32_max = (y_ref_f32 - y_vol_f32).abs().max()
    vollo_fp32_avg = (y_ref_f32 - y_vol_f32).abs().sum()
    print(f"Vollo fp32 error: {vollo_fp32_max:.5e}, {vollo_fp32_avg:.5e}")

    # === Final comparison

    max_improvement = (vollo_max / vollo_fp32_max).log2()
    avg_improvement = (vollo_avg / vollo_fp32_avg).log2()
    print(f"Max improvement: {max_improvement:.2f} bits")
    print(f"Avg improvement: {avg_improvement:.2f} bits")

    assert avg_improvement > 9.5
    assert max_improvement > 10.5

    assert False, "OK"
