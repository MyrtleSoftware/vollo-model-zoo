from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn


class _SELU(nn.Module):
    """
    Exactly `nn.SELU`, built from ops Vollo supports. Replace with `nn.SELU` once
    Vollo supports it natively.

    `scale * (relu(x) + alpha * (exp(min(x, 0)) - 1))`, rearranged to the form
    that measured fastest: `scale * alpha * (relu(x) / alpha + min(exp(x), 1) - 1)`.
    For x > 0 the clamped exponential is 1 and cancels the -1; for x <= 0 the
    ReLU is 0. Vollo fuses the clamp into the exp, the 1/alpha into the ReLU,
    and the outer `scale * alpha` and -1 into the next Linear, which leaves
    three pointwise instructions (ReLU, exp, add). A scale on the exp's output
    would cost a fourth.
    """

    _ALPHA = 1.6732632423543772848170429916717
    _SCALE = 1.0507009873554804934193349852946

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, sa = self._ALPHA, self._SCALE * self._ALPHA
        return sa * (torch.relu(x) / a + torch.clamp(torch.exp(x), max=1.0) - 1.0)


@beartype
def _round_robin_rows(rows: int, block_size: int, num_cores: int) -> list[int]:
    """
    Row counts of the first layer's pieces: two or more rounds of one piece per
    core, each piece at least two blocks (a Linear with one output block runs
    at half rate on the Agilex boards), and each core's total the compiler's
    own whole-block share of the layer. One piece on one core, or if that
    leaves one round.
    """
    blocks = -(-rows // block_size)
    rounds = blocks // (2 * num_cores)
    if num_cores == 1 or rounds < 2:
        return [rows]
    per_core = [
        blocks // num_cores + (c < blocks % num_cores) for c in range(num_cores)
    ]
    pieces = [
        n // rounds + (rounds - 1 - r < n % rounds)
        for r in range(rounds)
        for n in per_core
    ]
    ends = [min(rows, sum(pieces[: i + 1]) * block_size) for i in range(len(pieces))]
    return [e - s for s, e in zip([0, *ends], ends)]


class ReadoutFNN(nn.Module):
    @beartype
    def __init__(
        self,
        n_inputs: int,
        hidden: tuple[int, ...],
        n_outputs: int,
        selu: bool,
        num_cores: int = 1,
        block_size: int = 32,
    ):
        """
        A fully-connected network on the raw digitised I/Q trace of a readout
        window: the large readout networks of Lienhard et al.
        (https://arxiv.org/abs/2102.12481), HERQULES
        (https://arxiv.org/abs/2212.03895), Mude et al.
        (https://arxiv.org/abs/2405.08982) and the KLiNQ teachers
        (https://arxiv.org/abs/2503.03544). Joint networks read the whole
        multiplexed trace and emit one logit per joint state of all the qubits,
        which captures readout crosstalk that per-qubit filters miss.

        With `num_cores` > 1 the first Linear is split by output rows into
        pieces pinned to the cores round-robin, two or more pieces per core
        (`_round_robin_rows`). Left whole, the compiler gives each core one
        contiguous run of the hidden units, and the second layer, which reads
        them in order, waits for each core's run to finish. Interleaved, the
        first round of pieces lands on every core early and the second layer
        starts on it while the cores compute the next round. Each output is
        computed exactly as before; `load_reference` loads plain `nn.Linear`s.

        Args:
            n_inputs:   Trace length, I and Q interleaved (500 MS/s ADC: 2 ns each)
            hidden:     Hidden-layer widths
            n_outputs:  Joint states (2**qubits, or 3**qubits with leakage), or 1
            selu:       SELU as in Lienhard et al., or ReLU, which Vollo runs natively
            num_cores:  Cores to interleave the first layer over (1: leave it whole)
            block_size: The config's block size, so pieces are whole blocks
        """
        super().__init__()
        self.num_cores = num_cores
        rows = _round_robin_rows(hidden[0], block_size, num_cores)
        self.first = nn.ModuleList(nn.Linear(n_inputs, r) for r in rows)
        self.act = _SELU() if selu else nn.ReLU()
        layers: list[nn.Module] = []
        for d_in, d_out in zip(hidden, hidden[1:]):
            layers += [nn.Linear(d_in, d_out), _SELU() if selu else nn.ReLU()]
        layers.append(nn.Linear(hidden[-1], n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, n_inputs!]

        Returns:
            Tensor of shape [B, n_outputs!], logits (argmax on the controller)
        """
        pieces = []
        split = len(self.first) > 1
        for i, linear in enumerate(self.first):
            with (
                vollo_torch.CorePartition([i % self.num_cores])
                if split
                else nullcontext()
            ):
                pieces.append(linear(x))  # [B, rows_i!]
        h = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        return self.net(self.act(h))  # [B, n_outputs!]

    @torch.no_grad()
    def load_reference(self, linears: list[nn.Linear]):
        """Load the plain network's Linears, in order (as in an `nn.Sequential`)."""
        start = 0
        for piece in self.first:
            end = start + piece.out_features
            piece.weight.copy_(linears[0].weight[start:end])
            piece.bias.copy_(linears[0].bias[start:end])
            start = end
        rest = [m for m in self.net if isinstance(m, nn.Linear)]
        for linear, ref in zip(rest, linears[1:], strict=True):
            linear.load_state_dict(ref.state_dict())
        return self


class ReadoutTeachers(nn.Module):
    @beartype
    def __init__(self, n_qubits: int, n_inputs: int, hidden: tuple[int, ...]):
        """
        KLiNQ's teachers (https://arxiv.org/abs/2503.03544): one `ReadoutFNN`
        per qubit, each reading the whole multiplexed trace and giving that
        qubit's logit, so each qubit is classified independently (mid-circuit
        measurement). KLiNQ deploys only small distilled students on its FPGA;
        Vollo runs the teachers themselves.

        Args:
            n_qubits: Qubits, one teacher each
            n_inputs: Trace length, I and Q interleaved
            hidden:   Hidden-layer widths of each teacher
        """
        super().__init__()
        self.teachers = nn.ModuleList(
            ReadoutFNN(n_inputs, hidden, 1, selu=False) for _ in range(n_qubits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, n_inputs!]

        Returns:
            Tensor of shape [B, n_qubits!], one logit per qubit
        """
        return torch.cat([teacher(x) for teacher in self.teachers], dim=-1)


@beartype
def _vm(
    n_samples: int,
    hidden: tuple[int, ...],
    n_outputs: int,
    selu: bool,
    config: str,
    teachers: int = 0,
):
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    n_inputs = 2 * n_samples
    if teachers:
        model = ReadoutTeachers(teachers, n_inputs, hidden)
    else:
        # Interleaving measured 0 to -7% on the 6-core configs, but slower on
        # the 3-core IA-840f's SELU entries, so that board keeps it whole.
        hw = CONFIGS[config]
        cores = hw.num_cores if hw.block_size == 32 else 1
        model = ReadoutFNN(n_inputs, hidden, n_outputs, selu, cores, hw.block_size)
    input = torch.randn(1, n_inputs)

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=None,  # The decision needs the whole window, not causal
        meta=dict(
            window_ns=2 * n_samples,
            hidden=hidden,
            outputs=n_outputs,
            activation="SELU" if selu else "ReLU",
            teachers=teachers,
        ),
    )


@beartype
def main(config: str = "V80") -> Generator:
    # A 1 us window is 500 samples each of I and Q, interleaved. Hand-built FPGA
    # designs do not hold these networks: as HLS, HERQULES's baseline needs 217% to
    # 469% of a Zynq xczu7ev's LUTs, and Mude et al.'s about 420%. ReLU where
    # the paper names no activation.
    for x in [
        # HERQULES's baseline (Maurya et al.), 1000-500-250-32
        dict(n_samples=500, hidden=(500, 250), n_outputs=32, selu=False),
        # Mude et al.'s three-level baseline: 3**5 = 243 joint states with leakage
        dict(n_samples=500, hidden=(500, 250), n_outputs=243, selu=False),
        # ~1M baseline, Lienhard et al.'s rule at 400 samples (800 inputs):
        #   800*800+800 + 800*400+400 + 400*200+200 + 200*32+32 = 1,047,832
        dict(n_samples=400, hidden=(800, 400, 200), n_outputs=32, selu=True),
        # Lienhard et al.: 1000-1000-500-250-32, 1,634,782 parameters. ReLU is
        # paired with the paper's SELU to show what the rebuilt activation costs.
        dict(n_samples=500, hidden=(1000, 500, 250), n_outputs=32, selu=False),
        dict(n_samples=500, hidden=(1000, 500, 250), n_outputs=32, selu=True),
        # KLiNQ's five teachers, 1000-1000-500-250-1 each: 8,135,005 parameters
        dict(
            n_samples=500, hidden=(1000, 500, 250), n_outputs=1, selu=False, teachers=5
        ),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
