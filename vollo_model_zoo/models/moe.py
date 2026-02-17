from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


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

        self.router = nn.Linear(dim, 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, dim)

        Returns:
            x: Tensor of shape (Batch, Time, dim)
        """
        residual = x
        x = self.norm(x)

        x = self.router(x)

        return x + residual


@beartype
def _vm_moe(dim: int, hidden_dim: int, config: str):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 5, dim)

    model = MoE(dim=dim, hidden_dim=hidden_dim)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, hidden_dim=32 * 6 * 4),
    ]:
        yield _vm_moe(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
