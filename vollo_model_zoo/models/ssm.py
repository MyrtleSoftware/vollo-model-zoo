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
def _vm(
    dim: int,
    hidden: int,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, dim)

    model = SSM(dim=dim, hidden=hidden)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            hidden=hidden,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 12, hidden=32 * 12),
        dict(dim=32 * 18, hidden=32 * 14),
        dict(dim=32 * 6 * 4, hidden=32 * 6 * 8),
        dict(dim=32 * 6 * 8, hidden=32 * 6 * 16),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
