# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "torch", "vollo-compiler", "vollo-torch"]
# ///


import torch
import vollo_compiler
from beartype import beartype
from torch import nn
from vollo_torch.nn import PaddedConv1d


class MobileNet1D(nn.Module):
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

        def scale(v: int) -> int:
            return int(v * width_mult)

        # Roughly matching the paper (less the strides)

        self.features = nn.Sequential(
            _ConvNormAct(in_channels, scale(32), kernel_size=3),
            # Scale up
            _DepthwiseSeparableConv1d(scale(32), scale(64)),
            _DepthwiseSeparableConv1d(scale(64), scale(128)),
            _DepthwiseSeparableConv1d(scale(128), scale(128)),
            _DepthwiseSeparableConv1d(scale(128), scale(256)),
            _DepthwiseSeparableConv1d(scale(256), scale(256)),
            _DepthwiseSeparableConv1d(scale(256), scale(512)),
            # 5x block
            _DepthwiseSeparableConv1d(scale(512), scale(512)),
            _DepthwiseSeparableConv1d(scale(512), scale(512)),
            _DepthwiseSeparableConv1d(scale(512), scale(512)),
            _DepthwiseSeparableConv1d(scale(512), scale(512)),
            _DepthwiseSeparableConv1d(scale(512), scale(512)),
            # Final
            _DepthwiseSeparableConv1d(scale(512), scale(1024)),
            _DepthwiseSeparableConv1d(scale(1024), scale(1024)),
        )

        # In standard MobileNet there is avg pool + linear.
        # For 1D/Streaming, we just predict a class at each time step.
        self.classifier = nn.Linear(scale(1024), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        x = self.features(x)
        x = self.classifier(x.transpose(1, 2))
        return x


class _ConvNormAct(nn.Module):
    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = PaddedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
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
    ):
        super().__init__()
        self.depthwise = _ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            groups=in_channels,
        )
        self.pointwise = _ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@beartype
def instance(
    width_mult: float,
    config: vollo_compiler.Config,
) -> dict:
    in_channels = 3
    input_tensor = torch.randn(1, in_channels, 4)

    model = MobileNet1D(in_channels=in_channels, width_mult=width_mult)

    return dict(
        model=model,
        x=input_tensor,
        time_axis=2,
        meta=dict(
            width_mult=width_mult,
        ),
    )


SIZES = [
    dict(width_mult=0.1),
    dict(width_mult=0.42),
    dict(width_mult=1.0),
]


if __name__ == "__main__":
    # Standalone: `uv run <this file>` prints each size's VM latency on a V80.
    import vollo_torch as vt

    config = vollo_compiler.Config.amd_v80_c6b32()
    print(f"VM estimates on {config.name}:")
    for size in SIZES:
        features = size.get("_features", [])
        if "_experimental" in size or not set(features) <= set(config.features):
            continue
        kwargs = {k: v for k, v in size.items() if not k.startswith("_")}
        args = instance(**kwargs, config=config)
        model, _ = vt.fx.prepare_shape(args["model"], args["x"])
        nnir = vt.fx.nnir.to_nnir(model)
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
            print(f"\t{kwargs}: too big for {config.name} ({type(e).__name__})")
            continue
        params = sum(p.numel() for p in args["model"].parameters())
        cycles = program.cycle_count_per_inference()
        latency = program.compute_duration_per_inference_us(spaced=True)
        print(f"\t{kwargs}: {params} params, {cycles} cycles, {latency:.3f}us")
