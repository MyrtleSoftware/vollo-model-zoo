import torch
from beartype import beartype

from vollo_model_zoo.approx import round_mantisa
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


from matplotlib import pyplot as plt
from vollo_torch import Fp32Activations


def fp32sig(x):
    with Fp32Activations():
        denom = 1.0 + torch.exp(-x)

    # Now we want to compute 1 / denom, this is a bf16 approx
    y = 1 / denom

    with Fp32Activations():
        # do newton-raphson to compute 1 / denom in fp32
        for _ in range(2):
            y = y * (2 - denom * y)

        return y


def test_meta():
    x = gen_all_bf16().to(torch.float32)

    # Get region of interest (saturation of sigmoid)
    b = 20
    x = x[-b < x]
    x = x[x < b]

    y_ref_f32 = torch.sigmoid(x)

    assert y_ref_f32.isfinite().all()

    for n in [16]:
        y_ref_bn = round_mantisa(y_ref_f32, n=n - 9)
        ref_max = (y_ref_f32 - y_ref_bn).abs().max()
        ref_avg = (y_ref_f32 - y_ref_bn).abs().sum()
        print(f"Ideal bf{n} error: {ref_max:.5e}, {ref_avg:.5e}")

    # === bf16 sigmoid

    y_vol_b16 = vollo_fn(torch.sigmoid, "V80")(x).to(torch.float32)

    assert y_vol_b16.isfinite().all()

    vollo_max = (y_ref_f32 - y_vol_b16).abs().max()
    vollo_avg = (y_ref_f32 - y_vol_b16).abs().sum()

    print(f"Vollo bf16 error: {vollo_max:.5e}, {vollo_avg:.5e}")

    # === fp32 sigmoid

    y_vol_fp32 = vollo_fn(fp32sig, "V80")(x).to(torch.float32)

    assert y_vol_fp32.isfinite().all()

    vollo_fp32_max = (y_ref_f32 - y_vol_fp32).abs().max()
    vollo_fp32_avg = (y_ref_f32 - y_vol_fp32).abs().sum()

    print(f"Vollo fp32 error: {vollo_fp32_max:.5e}, {vollo_fp32_avg:.5e}")

    assert False, "ok"
