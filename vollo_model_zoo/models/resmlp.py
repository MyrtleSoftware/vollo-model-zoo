from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class _Aff(nn.Module):
    @beartype
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class _GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximation, see: https://arxiv.org/pdf/1606.08415
        """
        return x * nn.functional.sigmoid(1.702 * x)


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


class _CrossChannel(nn.Module):
    @beartype
    def __init__(self, dim: int, activation: str, expansion: int = 4):
        super().__init__()

        match activation.lower():
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = _GELU()
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


@beartype
def _vm(
    dim: int,
    classes: int,
    patches: int,
    layers: int,
    activation: str,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, patches, dim)

    model = ResMLP(
        dim=dim,
        num_classes=classes,
        num_patches=patches,
        num_layers=layers,
        activation=activation,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=None,  # Not a streaming model
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            patches=patches,
            layers=layers,
            activation=activation,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=64, classes=10, patches=6**2, layers=4, activation="ReLU"),
        dict(dim=64, classes=10, patches=6**2, layers=4, activation="GELU"),
        dict(dim=160, classes=10, patches=3**2, layers=5, activation="ReLU"),
        dict(dim=160, classes=10, patches=3**2, layers=5, activation="GELU"),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
