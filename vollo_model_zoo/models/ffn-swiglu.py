from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class LlamaSwiGLU(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """
        Transformer++ FFN block with SwiGLU activation and RMSNorm.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm = nn.RMSNorm(dim, eps=1e-5)

        # Fused linear layer for gate and value
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, dim)

        Returns:
            x: Tensor of shape (Batch, Time, dim)
        """
        residual = x
        x = self.norm(x)

        # Fused linear for gate and value
        xg = self.w13(x)
        x = xg[..., : self.hidden_dim]
        gate = xg[..., self.hidden_dim :]
        x = torch.nn.functional.silu(gate) * x

        x = self.w2(x)
        return x + residual


@beartype
def _vm_ffn_swiglu(dim: int, hidden_dim: int, config: str):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 5, dim)

    model = LlamaSwiGLU(dim=dim, hidden_dim=hidden_dim)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
            activation="SwiGLU",
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, hidden_dim=32 * 6 * 4),
        dict(dim=48 * 6, hidden_dim=48 * 6 * 4),
        dict(dim=64 * 6, hidden_dim=64 * 6 * 4),
    ]:
        yield _vm_ffn_swiglu(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
