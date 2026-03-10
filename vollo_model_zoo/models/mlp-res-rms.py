from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class MLPResRMS(nn.Module):
    @beartype
    def __init__(
        self,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
    ):
        """
        MLP with residuals and RMSNorm (Pre-norm).

        Args:
               num_layers:           Number of residual blocks
               dim:                  Input/output dimension
               hidden_dim:           Hidden dimension within each block
               activation:           Activation function to use
               bias:                 Whether to use bias in the linear layers
        """
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                _MLPResRMSBlock(dim, hidden_dim, activation, bias=bias)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, dim)

        Returns:
            x: Tensor of shape (Batch, Time, dim)
        """
        return self.blocks(x)


class _MLPResRMSBlock(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.ffn1 = nn.Linear(dim, hidden_dim, bias=bias)

        if activation.lower() not in ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation} not in {list(ACTIVATIONS.keys())}"
            )

        self.act = ACTIVATIONS[activation.lower()]()
        self.ffn2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: x = x + MLP(RMSNorm(x))
        residual = x
        x = self.norm(x)
        x = self.ffn1(x)
        x = self.act(x)
        x = self.ffn2(x)
        return x + residual


class _ELU(nn.Module):
    """
    Not supported natively in Vollo
    """

    def forward(self, x: torch.Tensor, alpha=1.0) -> torch.Tensor:
        return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "elu": _ELU,
}


@beartype
def _vm(
    num_layers: int,
    dim: int,
    hidden_dim: int,
    activation: str,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 5, dim)

    model = MLPResRMS(
        num_layers=num_layers,
        dim=dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            activation=activation,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for dim, hidden in [(320, 768), (512, 1024)]:
        # Use the same size but vary the activation function
        size_params = dict(num_layers=2, dim=dim, hidden_dim=hidden)

        for activation in ACTIVATIONS.keys():
            yield _vm(**size_params, activation=activation, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
