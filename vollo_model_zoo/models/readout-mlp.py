from collections.abc import Generator
from itertools import accumulate
from pathlib import Path

import torch
from beartype import beartype
from torch import nn


class _SELU(nn.Module):
    """
    Exactly `nn.SELU`, built from ops Vollo supports. Replace with `nn.SELU` once
    Vollo supports it natively.

    `scale * (relu(x) + alpha * (exp(min(x, 0)) - 1))`, rearranged to the form
    that measured fastest on V80: min(scale * alpha * exp(x), scale * alpha) is
    the negative branch plus its offset, so only one clamp and one exp sit in
    the dependency chain.
    """

    _ALPHA = 1.6732632423543772848170429916717
    _SCALE = 1.0507009873554804934193349852946

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = self._SCALE * self._ALPHA
        return self._SCALE * torch.relu(x) + torch.clamp(sa * torch.exp(x), max=sa) - sa


class ReadoutFNN(nn.Module):
    @beartype
    def __init__(self, n_samples: int, n_qubits: int, selu: bool = True):
        """
        Joint readout of several qubits sharing one resonator line: a
        fully-connected network maps the raw digitised trace of the integration
        window to the state of all qubits at once, which captures readout
        crosstalk that per-qubit matched filters miss. Lienhard et al.,
        https://arxiv.org/abs/2102.12481: layers n, n, n/2, n/4 for n inputs,
        SELU, and one output per joint state.

        Args:
            n_samples: Samples per I and Q trace (500 MS/s ADC: 2 ns each)
            n_qubits:  Qubits on the line; the output has 2**n_qubits joint states
            selu:      The paper's SELU, or ReLU, which Vollo runs natively
        """
        super().__init__()
        n = 2 * n_samples
        widths = [n, n, n // 2, n // 4]

        layers: list[nn.Module] = []
        for d_in, d_out in zip(widths, widths[1:]):
            layers += [nn.Linear(d_in, d_out), _SELU() if selu else nn.ReLU()]
        layers.append(nn.Linear(widths[-1], 2**n_qubits))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 2 * n_samples!], I and Q interleaved

        Returns:
            Tensor of shape [B, 2**n_qubits!], joint-state logits (argmax on the
            controller)
        """
        return self.net(x)


class _Student(nn.Module):
    def __init__(self, n_in: int | None, n_features: int):
        super().__init__()
        # Interval-averaged I and Q, the matched-filter scalar and the
        # normalisation are all linear in the trace, so they fold exactly into
        # this one constant layer: n_features = 1 + 2 * (number of averaging windows).
        # n_in=None leaves it out: the features arrive precomputed.
        self.front_end = None if n_in is None else nn.Linear(n_in, n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x if self.front_end is None else self.front_end(x))


class QubitReadoutStudents(nn.Module):
    @beartype
    def __init__(
        self, n_samples: int, features: tuple[int, ...], front_end: bool = True
    ):
        """
        Independent per-qubit readout, so any one qubit can be measured at any
        time (mid-circuit measurement): the distilled student networks of KLiNQ,
        Guo et al., https://arxiv.org/abs/2503.03544, on the five-qubit dataset
        of Lienhard et al. Each student reads its own qubit's demodulated trace
        and emits one logit. Qubits 1, 4 and 5 use FNN-A, 31-16-8-1, whose 31
        features are the 15 + 15 means of I and Q over 32-sample (64 ns)
        windows plus one matched-filter scalar; qubits 2 and 3 use FNN-B,
        201-16-8-1, with 100 + 100 five-sample (10 ns) means plus the scalar.
        A 1 us window is 500 samples, so FNN-A ignores the last 20 (zero
        weights in its front end).

        With `front_end=False` the networks start at the features: the
        averaging and matched filter run upstream of Vollo, as the samples
        stream in (in RTL, as in KLiNQ's own FPGA design), and Vollo receives
        only the concatenated features and runs the five 16-8-1 networks.

        Args:
            n_samples: Samples per I and Q trace (500 for the 1 us window)
            features:  Per-qubit student input width: 31 (FNN-A) or 201 (FNN-B)
            front_end: Compute the features from the raw traces here (True), or
                take them precomputed from upstream (False)
        """
        super().__init__()
        self.students = nn.ModuleList(
            _Student(2 * n_samples if front_end else None, f) for f in features
        )
        # Where each student's features start and end in the concatenated input
        self.offsets = None if front_end else [0, *accumulate(features)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, n_qubits, 2 * n_samples!], each qubit's own
                demodulated trace, I and Q interleaved; or, with
                `front_end=False`, [B, sum(features)!], the students' features
                concatenated

        Returns:
            Tensor of shape [B, n_qubits!], one logit per qubit (threshold on the
            controller)
        """
        if self.offsets is None:
            xs = [x[:, q] for q in range(len(self.students))]
        else:
            xs = [x[:, a:b] for a, b in zip(self.offsets, self.offsets[1:])]
        return torch.cat([student(xq) for student, xq in zip(self.students, xs)], -1)


@beartype
def _vm(model: nn.Module, input: torch.Tensor, config: str, **meta):
    from vollo_model_zoo.vm import vollo_info

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=None,  # The decision needs the whole window, not causal
        meta=meta,
    )


@beartype
def main(config: str = "V80") -> Generator:
    n_samples, n_qubits = 500, 5
    window_ns = 2 * n_samples
    features = (31, 201, 201, 31, 31)
    # Paired with the next entry: same students, but the front end runs upstream
    # of Vollo, so only the 31 + 201 + 201 + 31 + 31 = 495 features go in.
    yield _vm(
        QubitReadoutStudents(n_samples, features, front_end=False),
        torch.randn(1, sum(features)),
        config,
        window_ns=window_ns,
        network="KLiNQ students, features in",
    )
    bank_in = torch.randn(1, n_qubits, 2 * n_samples)
    yield _vm(
        QubitReadoutStudents(n_samples, features),
        bank_in,
        config,
        window_ns=window_ns,
        network="KLiNQ students",
    )

    # Baseline, n_samples=400, n=800:
    #   800*800+800 + 800*400+400 + 400*200+200 + 200*32+32 = 1,047,832 parameters
    # The paper's size is n_samples=500, n=1000: 1,634,782 parameters.
    for n, selu in [(400, True), (500, False), (500, True)]:
        yield _vm(
            ReadoutFNN(n, n_qubits, selu),
            torch.randn(1, 2 * n),
            config,
            window_ns=2 * n,
            network="Lienhard FNN",
            activation="SELU" if selu else "ReLU",
        )


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
