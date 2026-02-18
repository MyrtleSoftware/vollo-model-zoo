from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class _Expert(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # nn.ReLU(),
        # nn.Linear(hidden_dim, dim, bias=bias),
        # )
        #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x)


class MoE(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """
        A model

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (before SwiGLU)
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        # self.hidden_dim = hidden_dim

        self.norm = nn.RMSNorm(dim, eps=1e-5)

        self.init = nn.Linear(dim, dim, bias=bias)

        self.router = nn.Linear(dim, 1, bias=bias)

        self.expert1 = _Expert(dim, dim)
        self.expert2 = _Expert(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, dim)

        Returns:
            x: Tensor of shape (Batch, dim)
        """
        residual = x
        x = self.norm(x)

        x = self.init(x)  # [!h]

        route = self.router(x)  # [!1]

        # print(route.shape)
        # print(self.expert1.w1.weight.shape)

        # w1 = torch.where(
        #     route[None, :] < 0, self.expert1.w1.weight, self.expert2.w1.weight
        # )

        y = self.expert2(x)

        print(x.shape)  # [h!]
        print(y.shape)  # [h h!]

        # [o! h] @ [h!]

        # x = w1 @ x

        w1 = self.expert1.w1.weight
        w2 = self.expert2.w1.weight

        mask = torch.where(route[:, None] < 0, w1, w2)

        z = mask @ x

        return z


@beartype
def _vm_moe(dim: int, hidden_dim: int, config: str):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(dim)

    model = MoE(dim=dim, hidden_dim=hidden_dim)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=None,
        allow_dynamic_weights=True,
        meta=dict(
            dim=dim,
            hidden=hidden_dim,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(dim=32 * 6, hidden_dim=32 * 6 * 4),
    ]:
        yield _vm_moe(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"	{result}")
