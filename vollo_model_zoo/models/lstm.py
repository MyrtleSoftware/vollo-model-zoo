from collections.abc import Generator
from pathlib import Path

import torch
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
def _vm(
    num_layers: int,
    hidden_size: int,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    # Time dimension is 1 because batch_first=True
    input = torch.randn(1, 5, hidden_size)

    model = LSTM(
        num_layers=num_layers,
        input_size=hidden_size,
        hidden_size=hidden_size,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            layers=num_layers,
            hidden_size=hidden_size,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        # Tiny
        dict(num_layers=1, hidden_size=128),
        # ~1M parameters baseline: 4 * (H + H) * H = 8 * H^2. 8 * 354^2 approx 1M
        dict(num_layers=1, hidden_size=354),
        dict(num_layers=2, hidden_size=250),
        dict(num_layers=4, hidden_size=177),
        # Bigger
        dict(num_layers=3, hidden_size=768),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
