from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class MLP(nn.Module):
    @beartype
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        out_features: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """
        Multi-Layer Perceptron with ReLU non-linearities.

        Args:
               num_layers:           Total number of linear layers
               in_features:          Number of input features
               out_features:         Number of output features
               hidden_features:      Number of hidden features in each hidden layer
               bias:                 Whether to use bias in the linear layers
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("MLP must have at least one layer")

        layers = []
        last_dim = in_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, hidden_features, bias=bias))
            layers.append(nn.ReLU())
            last_dim = hidden_features
        layers.append(nn.Linear(last_dim, out_features, bias=bias))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, in_features)

        Returns:
            x: Tensor of shape (Batch, Time, out_features)
        """
        return self.mlp(x)


@beartype
def instance(
    num_layers: int,
    hidden_features: int,
    config: str,
) -> dict:
    input = torch.randn(1, 5, hidden_features)

    model = MLP(
        num_layers=num_layers,
        in_features=hidden_features,
        out_features=hidden_features,
        hidden_features=hidden_features,
    )

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            layers=num_layers,
            n_features=hidden_features,
            activation="ReLU",
        ),
    )


SIZES = [
    dict(num_layers=2, hidden_features=512),
    dict(num_layers=7, hidden_features=384),
    dict(num_layers=4, hidden_features=512),
    dict(num_layers=3, hidden_features=1024),
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
