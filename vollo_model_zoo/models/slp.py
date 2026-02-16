from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


class SLP(nn.Module):
    @beartype
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        """
        Single Layer Perceptron with ReLU non-linearity.

        Args:
               in_features:          Number of input features
               out_features:         Number of output features
               activation:           Whether to use a ReLU activation
               bias:                 Whether to use bias in the convolutional layer
        """
        super().__init__()

        self.slp = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, in_features, Time)

        Returns:
            x: Tensor of shape (Batch, out_features, Time)
        """
        x = self.slp(x)
        x = torch.nn.functional.relu(x)
        return x


@beartype
def _vm_slp(in_features: int, out_features: int):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 5, in_features)

    model = SLP(in_features=in_features, out_features=out_features)

    return vollo_info(
        model,
        input,
        time_axis=1,
        layers=1,
        hidden=out_features,
        meta=dict(
            activation="ReLU",
        ),
    )


@beartype
def main() -> Generator:
    for x in [
        dict(in_features=128, out_features=128),
        dict(in_features=256, out_features=1024),
        dict(in_features=1024, out_features=1024),
    ]:
        yield _vm_slp(**x)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
