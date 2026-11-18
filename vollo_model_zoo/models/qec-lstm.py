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


class QECLSTM(nn.Module):
    @beartype
    def __init__(
        self,
        distance: int,
        hidden_size: int,
        soft: bool = False,
    ):
        """
        Streaming LSTM decoder for a rotated surface code of distance `distance`.

        Each inference consumes one syndrome round, a vector of `defects` (a
        stabiliser outcome XORed with the previous round's), and updates a
        2-layer LSTM; the head turns the recurrent state into the logit that the
        logical qubit has flipped. Follows Varbanov et al. 2023
        (https://arxiv.org/abs/2307.03280), after Baireuther et al. 2018
        (https://arxiv.org/abs/1705.07855) and 2019
        (https://arxiv.org/abs/1804.02926). The last round also carries the
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
        """
        super().__init__()
        stabilisers = distance**2 - 1
        self.num_defects = stabilisers * (2 if soft else 1)

        self.lstm = nn.LSTM(
            input_size=self.num_defects,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        # Two Linears summed rather than one on concat(state, final_defects): same
        # function, but no concatenation on the critical path
        self.state_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.final_defects_to_hidden = nn.Linear(
            stabilisers // 2, hidden_size, bias=False
        )
        self.logit = nn.Linear(hidden_size, 1)

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
        state, _ = self.lstm(defects)  # [B, T, hidden!]
        hidden = self.state_to_hidden(torch.relu(state))
        hidden = hidden + self.final_defects_to_hidden(final_defects)
        return self.logit(torch.relu(hidden))


@beartype
def instance(
    distance: int,
    hidden_size: int,
    soft: bool,
    config: vollo_compiler.Config,
) -> dict:
    model = QECLSTM(distance, hidden_size, soft=soft)
    num_inputs = model.num_defects + (distance**2 - 1) // 2
    input = torch.randn(1, 5, num_inputs)

    return dict(
        model=model,
        x=input,
        time_axis=1,
        meta=dict(
            distance=distance,
            hidden_size=hidden_size,
            soft=soft,
        ),
    )


SIZES = [
    # Paper sizes (Varbanov et al. 2023), hard defects, then soft inputs
    dict(distance=3, hidden_size=64, soft=False),
    dict(distance=3, hidden_size=64, soft=True),
    dict(distance=5, hidden_size=128, soft=False),
    dict(distance=5, hidden_size=128, soft=True),
    dict(distance=7, hidden_size=128, soft=False),
    dict(distance=9, hidden_size=128, soft=False),
    # ~1M parameters baseline, d=11 (120 stabilisers, 60 final defects):
    dict(distance=11, hidden_size=256, soft=False),
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
