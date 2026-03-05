import torch
from beartype import beartype

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

    # Now we want to compute 1 / denom

    y0 = 1 / denom  # bf16

    with Fp32Activations():
        # do newton-raphson to compute 1 / denom in fp32
        y1 = y0 * (2 - denom * y0)
        return y1


def test_meta():
    x = gen_all_bf16().to(torch.float32)

    # Get positive values only
    x = x[x > 0]

    y_ref_f32 = torch.sigmoid(x)
    y_ref_b16 = y_ref_f32.to(torch.bfloat16).to(torch.float32)
    y_vol_b16 = vollo_fn(torch.sigmoid, "V80")(x).to(torch.float32)

    assert y_ref_f32.isfinite().all()
    assert y_ref_b16.isfinite().all()
    assert y_vol_b16.isfinite().all()

    ref_max = (y_ref_f32 - y_ref_b16).abs().max()
    ref_avg = (y_ref_f32 - y_ref_b16).abs().mean()

    print(f"Intrinsic max error: {ref_max:.5e}")
    print(f"Intrinsic avg error: {ref_avg:.5e}")

    vollo_max = (y_ref_f32 - y_vol_b16).abs().max()
    vollo_avg = (y_ref_f32 - y_vol_b16).abs().mean()

    print(f"Vollo max error: {vollo_max:.5e}")
    print(f"Vollo avg error: {vollo_avg:.5e}")

    assert False, "ok"

    plt.plot(x, y_ref_f32.numpy(), label="Ref-f32")
    plt.plot(x, y_ref_b16.numpy(), label="Ref-b16")
    plt.plot(x, y_vol_b16.numpy(), label="Vollo")

    plt.xlim(-10, -3)
    plt.ylim(0, 0.02)

    plt.legend()
    plt.savefig("/home/conor/Downloads/f16.png")
