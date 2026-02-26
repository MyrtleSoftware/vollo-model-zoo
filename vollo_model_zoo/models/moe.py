from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class _Expert(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x)


class MoE(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """
        A model

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        # self.hidden_dim = hidden_dim

        self.norm = nn.RMSNorm(dim, eps=1e-5)

        self.n = 4
        self.router = nn.Linear(dim, 2**self.n, bias=bias)
        self.experts = torch.nn.ModuleList(
            _Expert(dim, hidden_dim) for _ in range(2**self.n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, dim)

        Returns:
            x: Tensor of shape (Batch, dim)
        """
        residual = x
        x = self.norm(x)

        route = self.router(x)  # [!n]

        xs = [self.experts[i](x) for i in range(len(self.experts))]

        for i, n in map(lambda i: (i, 2 ** (self.n - i)), range(self.n)):
            # Advanced indexing keeps the rank
            xs = [torch.where(route[[i]] < 0, xs[j], xs[j + 1]) for j in range(0, n, 2)]

        return xs[0] + residual


@beartype
def _vm_moe(dim: int, hidden_dim: int, config: str):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(dim)

    model = MoE(dim=dim, hidden_dim=hidden_dim)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=None,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    n = 9

    for x in [
        dict(dim=32 * n, hidden_dim=32 * n * 4),
    ]:
        yield _vm_moe(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
