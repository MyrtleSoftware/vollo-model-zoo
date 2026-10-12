from collections.abc import Generator
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class ContinuousQECLSTM(nn.Module):
    @beartype
    def __init__(self, hidden_size: int, samples_per_step: int, softmax: bool = False):
        """
        LSTM decoder for continuous quantum error correction of the 3-qubit
        bit-flip code, after Convy et al. (https://arxiv.org/abs/2110.10378).

        The two stabilisers (Z1Z2 and Z2Z3) are measured weakly and continuously,
        giving two noisy analog voltages sampled every 32 ns. A 2-layer LSTM
        tracks the posterior over the 2^3 = 8 basis states, from which a
        controller applies a bit-flip correction whenever the argmax changes.
        The same LSTM tracks qubit dynamics from weak measurement in
        https://arxiv.org/abs/2108.12023.

        Each sample is `[I1, I2, s0]`: the two voltages and s0, the initial basis
        state as a scalar 0-7, repeated every step. The paper only says "initial
        basis state"; the scalar encoding is inferred from its parameter count
        (hidden_size=32, samples_per_step=1 has 13,448 parameters, as in the
        paper's table).

        One Vollo inference costs far more than 32 ns, so `samples_per_step`
        consecutive samples are packed into each timestep's features: their 2k
        voltages plus s0 once. Packing changes the input the model is trained on:
        retrain it for each choice.

        Args:
            hidden_size:       LSTM hidden size (the paper uses 32)
            samples_per_step:  Consecutive samples packed into one timestep
            softmax:           Emit probabilities rather than logits. A controller
                               only needs the argmax, and the softmax adds latency.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2 * samples_per_step + 1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 8)
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, 2 * samples_per_step + 1)

        Returns:
            Tensor of shape (Batch, Time, 8)
        """
        x, _ = self.lstm(x)
        x = self.head(x)
        return torch.softmax(x, -1) if self.softmax else x


@beartype
def _vm(
    hidden_size: int,
    samples_per_step: int,
    softmax: bool,
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(1, 5, 2 * samples_per_step + 1)

    model = ContinuousQECLSTM(
        hidden_size=hidden_size,
        samples_per_step=samples_per_step,
        softmax=softmax,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            hidden_size=hidden_size,
            samples_per_step=samples_per_step,
            step_ns=32 * samples_per_step,
            softmax=softmax,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    # A step keeps up with the 32 ns sampling when latency_contiguous <= 32 ns * k:
    # k=1 never does (0.50 us), k=16 narrowly fails (0.513 us > 0.512 us on V80),
    # k=20 does (0.513 us < 0.640 us; 0.528 us on a 320 MHz IP core), as does k=32.
    for x in [
        # Paper size, per-sample steps: softmax costs 0.22 us over logits
        dict(hidden_size=32, samples_per_step=1, softmax=True),
        dict(hidden_size=32, samples_per_step=1, softmax=False),
        dict(hidden_size=32, samples_per_step=20, softmax=False),
        dict(hidden_size=32, samples_per_step=32, softmax=False),
        dict(hidden_size=64, samples_per_step=32, softmax=False),
        dict(hidden_size=128, samples_per_step=32, softmax=False),
        # ~1M parameters baseline, input 2 * 64 + 1 = 129 features:
        # 4 * 272 * (129 + 272) + 4 * 272 * (272 + 272) + 2 * 8 * 272 + 8 * 272 + 8
        # = 1,034,696. Needs k >= 38 to keep up (1.21 us), hence k=64.
        dict(hidden_size=272, samples_per_step=64, softmax=False),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
