from collections.abc import Generator
from pathlib import Path
from typing import Optional

import torch
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


class _ConvNormAct(nn.Module):
    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = PaddedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.RMSNorm(out_channels, eps=1e-5)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # RMSNorm expects (..., Channels), so transpose
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        return x


class _DepthwiseSeparableConv1d(nn.Module):
    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.depthwise = _ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = _ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):
    @beartype
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_mult: float = 1.0,
    ):
        """
        MobileNet v1 1D variant.
        """
        super().__init__()

        def c(v: int) -> int:
            return int(v * width_mult)

        self.features = nn.Sequential(
            _ConvNormAct(in_channels, c(32), kernel_size=3, stride=2),
            _DepthwiseSeparableConv1d(c(32), c(64), stride=1),
            _DepthwiseSeparableConv1d(c(64), c(128), stride=2),
            _DepthwiseSeparableConv1d(c(128), c(128), stride=1),
            _DepthwiseSeparableConv1d(c(128), c(256), stride=2),
            _DepthwiseSeparableConv1d(c(256), c(256), stride=1),
            _DepthwiseSeparableConv1d(c(256), c(512), stride=2),
            # 5x block
            _DepthwiseSeparableConv1d(c(512), c(512), stride=1),
            _DepthwiseSeparableConv1d(c(512), c(512), stride=1),
            _DepthwiseSeparableConv1d(c(512), c(512), stride=1),
            _DepthwiseSeparableConv1d(c(512), c(512), stride=1),
            _DepthwiseSeparableConv1d(c(512), c(512), stride=1),
            #
            _DepthwiseSeparableConv1d(c(512), c(1024), stride=2),
            _DepthwiseSeparableConv1d(c(1024), c(1024), stride=1),
        )

        # In standard MobileNet there is avg pool + linear.
        # For 1D/Streaming, we might keep it sequence-to-sequence or just feature extraction.
        # To match the reference "classifier" but in 1D, we can use a 1x1 conv
        # which acts as a Linear layer per timestep.
        self.classifier = nn.Conv1d(c(1024), num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@beartype
def _vm_mobilenet(
    width_mult: float,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    in_channels = 3
    input_tensor = torch.randn(1, in_channels, 1000)

    model = MobileNet(in_channels=in_channels, width_mult=width_mult)

    return vollo_info(
        model,
        input_tensor,
        config=config,
        time_axis=2,
        meta=dict(
            width_mult=width_mult,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(width_mult=0.25),
        dict(width_mult=0.5),
        dict(width_mult=0.75),
        dict(width_mult=1.0),
    ]:
        yield _vm_mobilenet(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
