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
            # Using: sigmoid(-x) = 1 - sigmoid(x), we prioritise accuracy of h
            # when z is close to 1.

            one_minus_z = torch.sigmoid(-z)

            with vollo_torch.Fp32Activations():
                h = one_minus_z * n + (1 - one_minus_z) * h
        else:
            z = torch.sigmoid(z)
            h = (1 - z) * n + z * h

        return h, h


@beartype
def instance(
    input_size: int,
    hidden_size: int,
    layers: int,
    fp32: bool,
    config: str,
) -> dict:
    input = torch.randn(1, input_size)

    model = GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=layers, fp32=fp32
    )

    return dict(
        model=model,
        x=input,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            fp32=fp32,
            input=input_size,
            hidden=hidden_size,
            layers=layers,
        ),
    )


SIZES = [
    dict(input_size=512, hidden_size=384, layers=1, fp32=False),
    dict(input_size=512, hidden_size=384, layers=1, fp32=True),
    dict(input_size=512, hidden_size=512, layers=1, fp32=False),
    dict(input_size=512, hidden_size=512, layers=1, fp32=True),
    dict(input_size=512, hidden_size=512, layers=3, fp32=False),
    dict(input_size=512, hidden_size=512, layers=3, fp32=True),
]


@beartype
def main(config: str = "V80") -> Generator:
    from vollo_model_zoo.vm import sweep, vollo_info

    for size in sweep(SIZES, config):
        yield vollo_info(**instance(**size, config=config), config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
