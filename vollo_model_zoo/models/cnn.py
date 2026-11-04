from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


class CNN(nn.Module):
    @beartype
    def __init__(
        self,
        num_layers: int,
        channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        """
        Standard CNN with residual blocks.

        Args:
               num_layers:           Number of CNN blocks
               channels:             Number of channels in hidden layers
               kernel_size:          Size of the convolutional kernel
               bias:                 Whether to use bias in the convolutional layers
        """
        super().__init__()

        kwargs = dict(
            channels=channels,
            kernel_size=kernel_size,
            bias=bias,
        )

        self.blocks = nn.Sequential(*[_CNNBlock(**kwargs) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, channels, Time)

        Returns:
            x: Tensor of shape (Batch, channels, Time)
        """
        return self.blocks(x)


class _CNNBlock(nn.Module):
    @beartype
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
    ):
        """
        Pre-normed CNN block with residual connection.
        """
        super().__init__()

        self.norm = nn.RMSNorm(channels, eps=1e-5)

        self.conv = PaddedConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=bias,
        )

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Time)
        """

        residual = x

        # We transpose to (Batch, Time, Channels) for RMSNorm
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.act(x)

        x = x + residual

        return x


@beartype
def instance(
    num_layers: int,
    channels: int,
    kernel_size: int,
    config: str,
) -> dict:
    input = torch.randn(1, channels, 1000)

    model = CNN(
        num_layers=num_layers,
        channels=channels,
        kernel_size=kernel_size,
    )

    return dict(
        model=model,
        x=input,
        time_axis=2,
        meta=dict(
            layers=num_layers,
            channels=channels,
            kernel_size=kernel_size,
        ),
    )


SIZES = [
    dict(num_layers=4, channels=64, kernel_size=8),
    dict(num_layers=4, channels=64, kernel_size=32),
    dict(num_layers=4, channels=64, kernel_size=64),
    dict(num_layers=4, channels=256, kernel_size=32),
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
