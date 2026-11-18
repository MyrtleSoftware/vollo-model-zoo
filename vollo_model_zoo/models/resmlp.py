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


class ResMLP(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_patches: int,
        num_layers: int,
        activation: str,
    ):
        """
        ResMLP architecture, see: https://arxiv.org/pdf/2105.03404

        Args:
            patch_dim:    Internal dimension (channels per patch)
            num_classes:  Number of output classes
            num_patches:  Number of patches (N^2)
        """
        super().__init__()

        self.num_patches = num_patches

        layers = []

        for _ in range(num_layers):
            layers.append(_CrossPatch(dim=dim, num_patches=num_patches))
            layers.append(_CrossChannel(dim=dim, activation=activation))

        self.blocks = nn.Sequential(*layers)

        self.out_proj = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Map a batch of N^2 patches, each a d-dimensional feature vector
        to an output class prediction.

        Args:
            x: Tensor of shape (Batch, N^2, d)

        Returns:
            x: Tensor of shape (Batch, num_classes)
        """
        x = self.blocks(x)

        # Global average pooling across patches
        x = x.sum(dim=1) / self.num_patches

        return self.out_proj(x)


class _CrossChannel(nn.Module):
    @beartype
    def __init__(self, dim: int, activation: str, expansion: int = 4):
        super().__init__()

        match activation.lower():
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            _Aff(dim),
            nn.Linear(dim, dim * expansion),
            act,
            nn.Linear(expansion * dim, dim),
            _Aff(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Within channels: [B N H!] -> [B N H!]
        """
        return x + self.net(x)


class _CrossPatch(nn.Module):
    @beartype
    def __init__(self, dim: int, num_patches: int):
        super().__init__()

        self.net = nn.Sequential(
            _Aff(dim),
            _Mixer(num_patches=num_patches),
            _Aff(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _Mixer(nn.Module):
    @beartype
    def __init__(self, num_patches: int):
        super().__init__()
        self.mixer = nn.Linear(num_patches, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Within patches: [B N H!] -> [B N H!]
        """

        x = x.transpose(1, 2)
        x = self.mixer(x)
        x = x.transpose(1, 2)

        return x


class _Aff(nn.Module):
    @beartype
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


@beartype
def instance(
    dim: int,
    classes: int,
    patches: int,
    layers: int,
    activation: str,
    config: vollo_compiler.Config,
) -> dict:
    input = torch.randn(1, patches, dim)

    model = ResMLP(
        dim=dim,
        num_classes=classes,
        num_patches=patches,
        num_layers=layers,
        activation=activation,
    )

    return dict(
        model=model,
        x=input,
        time_axis=None,  # Not a streaming model
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            patches=patches,
            layers=layers,
            activation=activation,
        ),
    )


SIZES = [
    dict(dim=64, classes=10, patches=6**2, layers=4, activation="ReLU"),
    dict(dim=64, classes=10, patches=6**2, layers=4, activation="GELU"),
    dict(dim=160, classes=10, patches=3**2, layers=5, activation="ReLU"),
    dict(dim=160, classes=10, patches=3**2, layers=5, activation="GELU"),
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
