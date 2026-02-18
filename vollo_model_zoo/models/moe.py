from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class _Expert(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # nn.ReLU(),
        # nn.Linear(hidden_dim, dim, bias=bias),
        # )
        #

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

        self.router = nn.Linear(dim, 1, bias=bias)

        self.expert1 = _Expert(dim, hidden_dim)
        self.expert2 = _Expert(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, dim)

        Returns:
            x: Tensor of shape (Batch, dim)
        """
        residual = x
        x = self.norm(x)

        route = self.router(x)  # [!1]

        w1 = self.expert1.w1.weight
        w2 = self.expert2.w1.weight

        e1 = self.expert1(x)
        e2 = self.expert2(x)

        z = torch.where(route[:] < 0, e1, e2)

        return z


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
        allow_dynamic_weights=True,
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
