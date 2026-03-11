from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn


class GRU(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=1, bias=True, fp32: bool = False
    ):
        """
        GRU (Gated Recurrent Unit) network.

        Args:
               num_layers:           Number of recurrent layers
               input_size:           The number of expected features in the input x
               hidden_size:          The number of features in the hidden state h
               bias:                 Whether to use bias in the LSTM layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.layers = nn.Sequential().extend(
            _Layer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=bias,
                fp32=fp32,
            )
            for i in range(num_layers)
        )

    def forward(self, x):
        """
        Input:
            x: [T, *, input_size]
        Return:
            y: [T, *, hidden_size]
        """
        return self.layers(x)


class _Layer(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int, fp32: bool, bias: bool):
        super().__init__()
        step = _Step(
            input_size=input_size, hidden_size=hidden_size, fp32=fp32, bias=bias
        )
        self.scan = vollo_torch.nn.Scan(step)
        self.h_0 = torch.nn.Buffer(torch.zeros(hidden_size), persistent=False)

    def forward(self, x):
        return self.scan(x, self.h_0, input_axis=0, output_axis=0)


class _Step(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int, fp32: bool, bias: bool):
        super().__init__()

        self.fp32 = fp32

        self.context = vollo_torch.Fp32Activations if fp32 else nullcontext

        self.linear_ih_r = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_ih_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_ih_n = nn.Linear(input_size, hidden_size, bias=bias)

        self.linear_hh_r = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_hh_n = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_hh_z = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        x, h -> y, h
        """
        r = self.linear_ih_r(x) + self.linear_hh_r(h)
        z = self.linear_ih_z(x) + self.linear_hh_z(h)
        n = self.linear_ih_n(x) + torch.sigmoid(r) * self.linear_hh_n(h)

        # Use vollo's bf16 activation functions
        n = torch.tanh(n)

        if self.fp32:
            # High precsision for the update gate (multiplies fp32 h)
            z = sigmoid_bf16_hi(z)
        else:
            z = torch.sigmoid(z)

        with self.context():
            h = (1 - z) * n + z * h

        return h, h


def sigmoid_bf16_hi(x):
    """
    Compute sigmoid to approximately bf26 precision for bf16 inputs. This
    function _is_ compilable with Vollo. It should be called _outside_ of the
    Fp32Activations context.

    WARNING: for general fp32 inputs this is only bf18 precision.
    """
    # Prevent exp -> inf overflow
    x = -torch.clamp(x, min=-20)

    with vollo_torch.Fp32Activations():
        # Precision of z is ~ bf25, this is effectively "full precision" for
        # bf16 inputs so the following reciprocal is exact. However, for fp32
        # inputs this has ~ 7 bits of error hence, when we do the Newton
        # iterations this compounds to ~14 bits of error which results in only
        # 18 bits of precision.
        z = 1.0 + torch.exp(x)

    return recip_f32(z)


def recip_f32(x):
    """
    Given an f32 input, x, compute 1/x to almost full precision. This should be
    called outside of the Fp32Activation context.
    """
    # Now we want to compute 1 / z, this is a bf16 approx hence, ~6 bits of
    # mantissa precision
    y = 1 / x

    with vollo_torch.Fp32Activations():
        # Newton-Raphson to compute reciprocal in fp32, converges quadratically
        # so 6 -> 12 -> 24 mantissa bits
        y = y * (2 - x * y)
        y = y * (2 - x * y)

    return y


@beartype
def _vm(
    input_size: int,
    hidden_size: int,
    layers: int,
    fp32: bool,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, input_size)

    model = GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=layers, fp32=fp32
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            fp32=fp32,
            input=input_size,
            hidden=hidden_size,
            layers=layers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(input_size=512, hidden_size=384, layers=1, fp32=False),
        dict(input_size=512, hidden_size=384, layers=1, fp32=True),
        dict(input_size=512, hidden_size=512, layers=1, fp32=False),
        dict(input_size=512, hidden_size=512, layers=1, fp32=True),
        dict(input_size=512, hidden_size=512, layers=3, fp32=False),
        dict(input_size=512, hidden_size=512, layers=3, fp32=True),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
