from dataclasses import dataclass

import torch
import vollo_compiler
import vollo_torch
from torch import nn
from vollo_torch.nn import PaddedConv1d


class _1x1Conv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        super().__init__()

        self.conv = PaddedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class _WaveNetBlock(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        bias: bool,
    ):
        super().__init__()

        self.dilation_channels = dilation_channels

        self.dilated_conv = PaddedConv1d(
            in_channels=residual_channels,
            out_channels=2 * dilation_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

        self.res_conv = _1x1Conv1d(
            in_channels=dilation_channels,
            out_channels=residual_channels,
            bias=bias,
        )

        self.skip_conv = _1x1Conv1d(
            in_channels=dilation_channels,
            out_channels=skip_channels,
            bias=bias,
        )

    def forward(self, x):
        """
        Following:
            https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py


                   |----------------------------------------|     *residual*
                   |                                        |
                   |    |-- conv -- tanh --|                |
        -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
                        |-- conv -- sigm --|     |
                                                1x1
                                                 |
        ---------------------------------------> + ------------->	*skip*
        """
        residual = x

        # Fused dilated convolution
        xg = self.dilated_conv(x)
        x = xg[:, : self.dilation_channels, :]
        g = xg[:, self.dilation_channels :, :]

        # Gated activation unit
        x = torch.tanh(x) * torch.sigmoid(g)

        # 1x1 for skip
        s = self.skip_conv(x)

        # 1x1 for residual
        x = self.res_conv(x)
        x = x + residual

        return x, s


class WaveNet(nn.Module):
    def __init__(
        self,
        layers: int = 4,
        blocks: int = 1,
        in_channels: int = 1,
        out_channels: int = 256,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 32,
        kernel_size: int = 2,
        bias: bool = True,
    ):
        """
        See the paper: https://arxiv.org/pdf/1609.03499

        Args:
               layers:               Number of layers in each block
               blocks:               Number of wavenet blocks of this model
               in_channels:          Number of input channels
               out_channels:         Number of output channels
               residual_channels:    Number of channels for the residual connection
               dilation_channels:    Number of channels for the dilated convolution
               skip_channels:        Number of channels for the skip connections
               kernel_size:          Size of the dilation kernel
               bias:                 Whether to use bias in the convolutional layers
        """
        super().__init__()

        self.start_conv = _1x1Conv1d(in_channels, residual_channels, bias=bias)

        self.blocks = nn.ModuleList()

        for _ in range(blocks):
            for i in range(layers):
                self.blocks.append(
                    _WaveNetBlock(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=2**i,
                        bias=bias,
                    )
                )

        self.end_conv = nn.Sequential(
            nn.ReLU(),
            _1x1Conv1d(skip_channels, skip_channels, bias=bias),
            nn.ReLU(),
            _1x1Conv1d(skip_channels, out_channels, bias=bias),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, in_channels, Time)

        Returns:
            x: Tensor of shape (Batch, out_channels, Time)
        """

        x = self.start_conv(x)

        x, acc = self.blocks[0](x)

        for i in range(1, len(self.blocks)):
            x, s = self.blocks[i](x)
            acc = acc + s

        return self.end_conv(acc)


@dataclass
class WaveNetResult:
    layers_x_blocks: int
    hidden: int
    param_count_m: float
    latency_us: float


def test_wavenet(hidden: int, layers: int, blocks: int) -> WaveNetResult:
    in_channels = 1
    out_channels = 3

    model = WaveNet(
        layers=layers,
        blocks=blocks,
        in_channels=in_channels,
        out_channels=out_channels,
        residual_channels=hidden,
        dilation_channels=hidden,
        skip_channels=hidden,
    )

    batch_size = 1
    sequence_length = 5
    input = torch.randn(batch_size, in_channels, sequence_length)
    model, _ = vollo_torch.fx.prepare_shape(model, input)
    nnir = vollo_torch.fx.nnir.to_nnir(model)

    nnir, _ = nnir.streaming_transform(2)

    program = nnir.to_program(vollo_compiler.Config.v80_c6b32())

    program.pack()

    latency = program.compute_duration_per_inference_us()
    param_count = sum(p.numel() for p in model.parameters())

    return WaveNetResult(
        layers_x_blocks=layers * blocks,
        hidden=hidden,
        param_count_m=param_count / 1e6,
        latency_us=latency,
    )


def main():
    scenarios = [
        # Requested
        dict(layers=4, blocks=1, hidden=80),
        # Intermediate 4-layer
        dict(layers=4, blocks=1, hidden=32 * 6 * 2),
        # Max n-layer
        dict(layers=1, blocks=1, hidden=32 * 6 * 10),
        dict(layers=2, blocks=1, hidden=32 * 6 * 7),
        dict(layers=4, blocks=1, hidden=32 * 6 * 5),
    ]

    for x in scenarios:
        print(test_wavenet(**x))


if __name__ == "__main__":
    main()
