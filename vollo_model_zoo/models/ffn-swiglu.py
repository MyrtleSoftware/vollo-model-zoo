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


class LlamaSwiGLU(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
        fuse: bool = True,
    ):
        """
        Transformer++ FFN block with SwiGLU activation and RMSNorm.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.fuse = fuse

        if fuse:
            self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, dim)

        Returns:
            x: Tensor of shape (Batch, Time, dim)
        """
        residual = x
        x = self.norm(x)

        if self.fuse:
            xg = self.w13(x)
            x = xg[..., : self.hidden_dim]
            gate = xg[..., self.hidden_dim :]
        else:
            gate = self.w3(x)
            x = self.w1(x)

        x = torch.nn.functional.silu(gate) * x
        x = self.w2(x)
        return x + residual


@beartype
def instance(
    dim: int, hidden_dim: int, fuse: bool, config: vollo_compiler.Config
) -> dict:
    input = torch.randn(1, 5, dim)

    model = LlamaSwiGLU(dim=dim, hidden_dim=hidden_dim, fuse=fuse)

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
            activation="SwiGLU",
            fused=fuse,
        ),
    )


SIZES = [
    dict(dim=32 * 6, hidden_dim=32 * 6 * 4, fuse=True),
    dict(dim=32 * 6, hidden_dim=32 * 6 * 4, fuse=False),
    dict(dim=48 * 6, hidden_dim=48 * 6 * 4, fuse=True),
    dict(dim=48 * 6, hidden_dim=48 * 6 * 4, fuse=False),
    dict(dim=64 * 6, hidden_dim=64 * 6 * 4, fuse=True),
    dict(dim=64 * 6, hidden_dim=64 * 6 * 4, fuse=False),
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
        params = sum(p.numel() for p in args["model"].parameters())
        row = {**kwargs, "params (M)": params / 1e6}
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
            row["cycles"] = f"too big ({type(e).__name__})"
        else:
            row["cycles"] = program.cycle_count_per_inference()
            row["compute latency (us)"] = program.compute_duration_per_inference_us(
                spaced=True
            )
        rows.append(row)
    print(tabulate(rows, headers="keys", floatfmt=".2f"))
