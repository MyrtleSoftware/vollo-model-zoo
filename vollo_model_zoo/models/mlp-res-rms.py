from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class Exponential(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "exp": Exponential,
    "softplus": nn.Softplus,
}


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


@beartype
def _vm_mlp_res_rms(
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
    # Use the same size but vary the activation function
    size_params = dict(num_layers=2, dim=512, hidden_dim=1024)

    for activation in ["relu", "sigmoid", "tanh", "exp", "softplus"]:
        yield _vm_mlp_res_rms(**size_params, activation=activation, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
