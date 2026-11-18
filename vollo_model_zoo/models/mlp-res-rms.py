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


class MLPResRMS(nn.Module):
    @beartype
    def __init__(
        self,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
    ):
        """
        MLP with residuals and RMSNorm (Pre-norm).

        Args:
               num_layers:           Number of residual blocks
               dim:                  Input/output dimension
               hidden_dim:           Hidden dimension within each block
               activation:           Activation function to use
               bias:                 Whether to use bias in the linear layers
        """
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                _MLPResRMSBlock(dim, hidden_dim, activation, bias=bias)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, dim)

        Returns:
            x: Tensor of shape (Batch, Time, dim)
        """
        return self.blocks(x)


class _MLPResRMSBlock(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.ffn1 = nn.Linear(dim, hidden_dim, bias=bias)

        if activation.lower() not in ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation} not in {list(ACTIVATIONS.keys())}"
            )

        self.act = ACTIVATIONS[activation.lower()]()
        self.ffn2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: x = x + MLP(RMSNorm(x))
        residual = x
        x = self.norm(x)
        x = self.ffn1(x)
        x = self.act(x)
        x = self.ffn2(x)
        return x + residual


class _ELU(nn.Module):
    """
    Not supported natively in Vollo
    """

    def forward(self, x: torch.Tensor, alpha=1.0) -> torch.Tensor:
        return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "elu": _ELU,
}


@beartype
def instance(
    num_layers: int,
    dim: int,
    hidden_dim: int,
    activation: str,
    config: vollo_compiler.Config,
) -> dict:
    input = torch.randn(1, 5, dim)

    model = MLPResRMS(
        num_layers=num_layers,
        dim=dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
            activation=activation,
        ),
    )


SIZES = [
    # The same two sizes, with each activation function
    dict(num_layers=2, dim=dim, hidden_dim=hidden, activation=activation)
    for dim, hidden in [(320, 768), (512, 1024)]
    for activation in ["relu", "sigmoid", "tanh", "softplus", "silu", "elu"]
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
