# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "tabulate", "torch", "vollo-compiler", "vollo-torch"]
# ///


import torch
import vollo_compiler
import vollo_torch
from beartype import beartype
from tabulate import tabulate
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
    config: vollo_compiler.Config,
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


if __name__ == "__main__":
    # Standalone: `uv run <this file>` prints each size's VM compute latency on a V80.
    config = vollo_compiler.Config.amd_v80_c6b32()
    print(f"VM estimates on {config.name}:")
    rows = []
    for size in SIZES:
        features = size.get("_features", [])
        if "_experimental" in size or not set(features) <= set(config.features):
            continue
        kwargs = {k: v for k, v in size.items() if not k.startswith("_")}
        args = instance(**kwargs, config=config)
        model, _ = vollo_torch.fx.prepare_shape(args["model"], args["x"])
        nnir = vollo_torch.fx.nnir.to_nnir(model)
        if args["time_axis"] is not None:
            nnir, _ = nnir.streaming_transform(args["time_axis"])
        try:
            program = nnir.to_program(
                config,
                allow_dynamic_weights=args.get("allow_dynamic_weights", False),
                quick_compile=args.get("quick_compile", False),
            )
            program.pack()
        except (vollo_compiler.AllocationError, vollo_compiler.SaveError) as e:
            cycles, latency = f"too big ({type(e).__name__})", None
        else:
            cycles = program.cycle_count_per_inference()
            latency = program.compute_duration_per_inference_us(spaced=True)
        params = sum(p.numel() for p in args["model"].parameters())
        rows.append(
            {
                "params (M)": params / 1e6,
                "compute cycles": cycles,
                "compute latency (us)": latency,
                **kwargs,
            }
        )
    print(tabulate(rows, headers="keys", floatfmt=".2f"))
