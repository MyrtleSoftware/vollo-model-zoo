from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn


class QECLSTM(nn.Module):
    @beartype
    def __init__(
        self,
        distance: int,
        hidden_size: int,
        soft: bool = False,
        lstm2_cores: list[int] | None = None,
        head_core: int | None = None,
    ):
        """
        Streaming LSTM decoder for a rotated surface code of distance `distance`.

        Each inference consumes one syndrome round, a vector of `defects` (a
        stabiliser outcome XORed with the previous round's), and updates a
        2-layer LSTM; the head turns the recurrent state into the logit that the
        logical qubit has flipped. Follows Varbanov et al. 2023
        (https://arxiv.org/abs/2307.03280), after Baireuther et al. 2018
        (https://arxiv.org/abs/1705.07855). The last round also carries the
        `final_defects` of the logical basis, recomputed from the data-qubit
        readout, which only the head sees; the controller feeds zeros in earlier
        rounds and reads the output after the last one. The paper's auxiliary
        head is a training-time regulariser, so it is left out here.

        With `soft=True` the inputs follow Ali et al. 2024
        (https://arxiv.org/abs/2403.00706): the defect probability from the
        analog readout, plus a leakage flag, for every stabiliser.

        Args:
               distance:             Code distance d: d^2 - 1 stabilisers
               hidden_size:          Size of the LSTM state (the paper uses 64
                                     for d=3 and 96-128 for d=5 and d=7)
               soft:                 Defect probabilities and leakage flags
                                     instead of hard defects
               lstm2_cores:          Run the second LSTM layer on these cores
                                     (`vollo_torch.CorePartition`) rather than
                                     letting the compiler place it
               head_core:            Run the head on this one core rather than
                                     letting the compiler spread it

        The core placements change where the work runs, not the function
        (see `_vm` for when they pay off). The layers are two 1-layer
        `nn.LSTM`s, rather than one 2-layer one, so that each can be placed: a
        2-layer `nn.LSTM`'s `weight_ih_l1` etc. are `lstm2.weight_ih_l0` etc.
        here (`tests/test_qec_lstm.py` converts a state dict).
        """
        super().__init__()
        stabilisers = distance**2 - 1
        self.num_defects = stabilisers * (2 if soft else 1)

        self.lstm1 = nn.LSTM(self.num_defects, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # Two Linears summed rather than one on concat(state, final_defects): same
        # function, but no concatenation on the critical path
        self.state_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.final_defects_to_hidden = nn.Linear(
            stabilisers // 2, hidden_size, bias=False
        )
        self.logit = nn.Linear(hidden_size, 1)
        self.lstm2_cores = lstm2_cores
        self.head_core = head_core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, num_defects + (d^2 - 1) / 2), the
               round's defects followed by its final defects

        Returns:
            Tensor of shape (Batch, Time, 1): logit that the logical qubit has
            flipped. Thresholding at 0 is the same decision as a sigmoid at 0.5,
            and saves a sigmoid per round.
        """
        defects = x[..., : self.num_defects]
        final_defects = x[..., self.num_defects :]
        state, _ = self.lstm1(defects)  # [B, T, hidden!]
        with _on_cores(self.lstm2_cores):
            state, _ = self.lstm2(state)  # [B, T, hidden!]
        with _on_cores(None if self.head_core is None else [self.head_core]):
            hidden = self.state_to_hidden(torch.relu(state))
            hidden = hidden + self.final_defects_to_hidden(final_defects)
            return self.logit(torch.relu(hidden))


def _on_cores(cores: list[int] | None):
    """
    Run what is inside on `cores`, or wherever the compiler puts it for None
    """
    return nullcontext() if cores is None else vollo_torch.CorePartition(cores)


@beartype
def _vm(
    distance: int,
    hidden_size: int,
    soft: bool,
    config: str,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    # A head of at most 4x4 weight blocks (hidden 128 on a block size of 32,
    # 256 on 64) runs faster on one core than spread over the cores: the spread
    # saves less matmul time than its cross-core transfers cost. The second
    # LSTM layer, on three cores including the head's, beats the compiler's
    # choice of four. Past 4x4 blocks the compiler's own placement is kept: no
    # fixed placement measured better at every hidden size from 160 to 256
    lstm2_cores, head_core = None, None
    if hidden_size <= 4 * CONFIGS[config].block_size:
        lstm2_cores, head_core = [0, 1, 2], 0

    model = QECLSTM(
        distance,
        hidden_size,
        soft=soft,
        lstm2_cores=lstm2_cores,
        head_core=head_core,
    )
    num_inputs = model.num_defects + (distance**2 - 1) // 2
    input = torch.randn(1, 5, num_inputs)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=1,
        meta=dict(
            distance=distance,
            hidden_size=hidden_size,
            soft=soft,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        # Paper sizes (Varbanov et al. 2023), hard defects, then soft inputs
        dict(distance=3, hidden_size=64, soft=False),
        dict(distance=3, hidden_size=64, soft=True),
        dict(distance=5, hidden_size=128, soft=False),
        dict(distance=5, hidden_size=128, soft=True),
        dict(distance=7, hidden_size=128, soft=False),
        dict(distance=9, hidden_size=128, soft=False),
        # ~1M parameters baseline, d=11 (120 stabilisers, 60 final defects):
        dict(distance=11, hidden_size=256, soft=False),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
