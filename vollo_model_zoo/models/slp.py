# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "torch", "vollo-compiler", "vollo-torch"]
# ///


import torch
import vollo_compiler
from beartype import beartype
from torch import nn


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
               bias:                 Whether to use bias in the linear layer
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
            x: Tensor of shape (Batch, Time, in_features)

        Returns:
            x: Tensor of shape (Batch, Time, out_features)
        """
        x = self.slp(x)
        x = torch.nn.functional.relu(x)
        return x


@beartype
def instance(
    in_features: int, out_features: int, config: vollo_compiler.Config
) -> dict:
    input = torch.randn(1, 5, in_features)

    model = SLP(in_features=in_features, out_features=out_features)

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            input=in_features,
            output=out_features,
            activation="ReLU",
        ),
    )


SIZES = [
    dict(in_features=128, out_features=128),
    dict(in_features=256, out_features=1024),
    dict(in_features=1024, out_features=1024),
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
