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


class SSM(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden: int):
        super().__init__()

        self.ssm = vollo_torch.nn.Scan(_StepSSM(dim, hidden))
        self.h0 = torch.nn.Buffer(torch.zeros(hidden), persistent=False)

    def forward(self, x):
        """
        Args:
            x: [time, dim]

        Returns:
            r: [time, dim]
        """

        return self.ssm(x, self.h0, input_axis=0, output_axis=0)  # [t, D]


class _StepSSM(nn.Module):
    @beartype
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()

        self.hidden = hidden

        self.A = nn.Linear(hidden, hidden, bias=bias)
        self.B = nn.Linear(dim, hidden, bias=bias)
        self.C = nn.Linear(hidden, dim, bias=bias)
        self.D = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, h):
        """
        x [dim    ]
        h [hidden ]
        """

        h = self.A(h) + self.B(x)
        y = self.C(h) + self.D(x)

        return y, h


@beartype
def instance(
    dim: int,
    hidden: int,
    config: vollo_compiler.Config,
) -> dict:
    input = torch.randn(1, dim)

    model = SSM(dim=dim, hidden=hidden)

    return dict(
        model=model,
        x=input,
        time_axis=0,
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            hidden=hidden,
        ),
    )


SIZES = [
    dict(dim=32 * 12, hidden=32 * 12),
    dict(dim=32 * 18, hidden=32 * 14),
    dict(dim=32 * 6 * 4, hidden=32 * 6 * 8),
    dict(dim=32 * 6 * 8, hidden=32 * 6 * 16),
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
