from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from beartype.typing import Generator
from torch import nn


class SSM(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden: int):
        super().__init__()

        self.ssm = vollo_torch.nn.Scan(_StepSSM(dim, hidden))
        self.h0 = torch.nn.Buffer(torch.zeros(hidden), persistent=False)

    def forward(self, x):
        """
        Args:
            x: [time, dim]

        Returns:
            r: [time, dim]
        """

        return self.ssm(x, self.h0, input_axis=0, output_axis=0)  # [t, D]


class _StepSSM(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()

        self.hidden = hidden

        self.A = nn.Linear(hidden, hidden, bias=bias)
        self.B = nn.Linear(dim, hidden, bias=bias)
        self.C = nn.Linear(hidden, dim, bias=bias)
        self.D = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, h):
        """
        x [dim    ]
        h [hidden ]
        """

        h = self.A(h) + self.B(x)
        y = self.C(h) + self.D(x)

        return y, h


@beartype
def instance(
    dim: int,
    hidden: int,
    config: str,
) -> dict:
    input = torch.randn(1, dim)

    model = SSM(dim=dim, hidden=hidden)

    return dict(
        model=model,
        x=input,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            hidden=hidden,
        ),
    )


SIZES = [
    dict(dim=32 * 12, hidden=32 * 12),
    dict(dim=32 * 18, hidden=32 * 14),
    dict(dim=32 * 6 * 4, hidden=32 * 6 * 8),
    dict(dim=32 * 6 * 8, hidden=32 * 6 * 16),
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
