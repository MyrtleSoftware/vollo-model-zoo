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


class MoE(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden_dim: int, log_n_experts: int):
        """
        An MoE layer that selects the top 1 of 2**log_n_experts for each input token.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        assert log_n_experts > 0, "log_n_experts must be greater than 0"

        self.n = log_n_experts
        self.router = nn.Linear(dim, 2**self.n, bias=False)
        self.experts = torch.nn.ModuleList(
            _Expert(dim, hidden_dim) for _ in range(2**self.n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (*, dim)

        Returns:
            x: Tensor of shape (*, dim)
        """
        route = self.router(x)  # [..., !n]

        xs = [self.experts[i](x) for i in range(len(self.experts))]

        for i in range(self.n):
            mask = route[..., i : i + 1] < 0
            xs = [torch.where(mask, xs[j], xs[j + 1]) for j in range(0, len(xs), 2)]

        return xs[0]


class _Expert(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x)


@beartype
def instance(
    dim: int, hidden_dim: int, log_n_experts: int, config: vollo_compiler.Config
) -> dict:
    input = torch.randn(1, 4, dim)

    model = MoE(dim=dim, hidden_dim=hidden_dim, log_n_experts=log_n_experts)

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
            n_experts=2**log_n_experts,
        ),
    )


EXPERIMENTAL = "Vollo's support for MoE is still being improved"

SIZES = [
    dict(dim=192, hidden_dim=640, log_n_experts=2),
    dict(dim=192, hidden_dim=640, log_n_experts=3),
    dict(dim=192, hidden_dim=640, log_n_experts=4),
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
