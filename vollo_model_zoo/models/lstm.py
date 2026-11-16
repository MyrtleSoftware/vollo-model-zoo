# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "torch", "vollo-compiler", "vollo-torch"]
# ///


import torch
import vollo_compiler
from beartype import beartype
from torch import nn


class LSTM(nn.Module):
    @beartype
    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        """
        Long Short-Term Memory (LSTM) network.

        Args:
               num_layers:           Number of recurrent layers
               input_size:           The number of expected features in the input x
               hidden_size:          The number of features in the hidden state h
               bias:                 Whether to use bias in the LSTM layers
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, input_size)

        Returns:
            x: Tensor of shape (Batch, Time, hidden_size)
        """
        x, _ = self.lstm(x)
        return x


@beartype
def instance(
    num_layers: int,
    hidden_size: int,
    config: vollo_compiler.Config,
) -> dict:
    # Time dimension is 1 because batch_first=True
    input = torch.randn(1, 5, hidden_size)

    model = LSTM(
        num_layers=num_layers,
        input_size=hidden_size,
        hidden_size=hidden_size,
    )

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            layers=num_layers,
            hidden_size=hidden_size,
        ),
    )


SIZES = [
    # Tiny
    dict(num_layers=1, hidden_size=128),
    # ~1M parameters baseline: 4 * (H + H) * H = 8 * H^2. 8 * 354^2 approx 1M
    dict(num_layers=1, hidden_size=354),
    dict(num_layers=2, hidden_size=250),
    dict(num_layers=4, hidden_size=177),
    # Bigger
    dict(num_layers=3, hidden_size=768),
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
