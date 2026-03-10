from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class MoE(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden_dim: int, log_n_experts: int):
        """
        An MoE layer that selects the top 1 of 2**log_n_experts for each input token.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        assert log_n_experts > 0, "log_n_experts must be greater than 0"

        self.n = log_n_experts
        self.router = nn.Linear(dim, 2**self.n, bias=False)
        self.experts = torch.nn.ModuleList(
            _Expert(dim, hidden_dim) for _ in range(2**self.n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (*, dim)

        Returns:
            x: Tensor of shape (*, dim)
        """
        route = self.router(x)  # [..., !n]

        xs = [self.experts[i](x) for i in range(len(self.experts))]

        for i in range(self.n):
            mask = route[..., i : i + 1] < 0
            xs = [torch.where(mask, xs[j], xs[j + 1]) for j in range(0, len(xs), 2)]

        return xs[0]


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


@beartype
def _vm(dim: int, hidden_dim: int, log_n_experts: int, config: str):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 4, dim)

    model = MoE(dim=dim, hidden_dim=hidden_dim, log_n_experts=log_n_experts)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
            n_experts=2**log_n_experts,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=192, hidden_dim=640, log_n_experts=2),
        dict(dim=192, hidden_dim=640, log_n_experts=3),
        dict(dim=192, hidden_dim=640, log_n_experts=4),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
