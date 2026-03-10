from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


class TCN(nn.Module):
    @beartype
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        bias: bool = True,
    ):
        """
        Temporal Convolutional Network (TCN).

        See the paper: https://arxiv.org/pdf/1803.01271

        Args:
               num_inputs:           Number of input channels
               num_channels:         List of channel sizes for each block
               kernel_size:          Size of the convolutional kernel
               bias:                 Whether to use bias in the convolutional layers
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                _TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    bias=bias,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, num_inputs, Time)

        Returns:
            x: Tensor of shape (Batch, num_channels[-1], Time)
        """
        return self.network(x)


class _TemporalBlock(nn.Module):
    @beartype
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
    ):
        super().__init__()

        self.conv1 = PaddedConv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.act1 = nn.ReLU()

        self.conv2 = PaddedConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.act2 = nn.ReLU()

        self.downsample = (
            _1x1Conv1d(n_inputs, n_outputs, bias=bias)
            if n_inputs != n_outputs
            else None
        )

        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, n_inputs, Time)
        """
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        return self.final_act(out + residual)


class _1x1Conv1d(nn.Module):
    @beartype
    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        super().__init__()

        self.conv = PaddedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@beartype
def _vm(
    num_inputs: int,
    num_channels: list[int],
    kernel_size: int,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, num_inputs, 10)

    model = TCN(
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=kernel_size,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=2,
        meta=dict(
            inputs=num_inputs,
            kernel=kernel_size,
            channels=str(num_channels),
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(num_inputs=1, num_channels=[128] * 3, kernel_size=3),
        dict(num_inputs=1, num_channels=[256] * 3, kernel_size=3),
        dict(num_inputs=1, num_channels=[512] * 3, kernel_size=3),
        dict(num_inputs=1, num_channels=[2**i for i in range(5, 11)], kernel_size=3),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
