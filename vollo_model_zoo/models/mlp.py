# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "torch", "vollo-compiler", "vollo-torch"]
# ///


import torch
import vollo_compiler
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
    config: vollo_compiler.Config,
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
