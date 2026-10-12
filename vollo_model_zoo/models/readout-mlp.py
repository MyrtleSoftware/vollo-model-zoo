import contextlib
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path

import torch
import torch.nn.functional as F
import vollo_torch
from beartype import beartype
from torch import nn


class _SELU(nn.Module):
    """
    `nn.SELU` built from ops Vollo supports. Replace with `nn.SELU` once Vollo
    supports it natively.

    `scale * (relu(x) + alpha * (exp(min(x, 0)) - 1))`, rearranged to the form
    that measured fastest: `scale * alpha * (relu(x) / alpha + min(exp(x), 1) - 1)`.
    For x > 0 the clamped exponential is 1 and cancels the -1; for x <= 0 the
    ReLU is 0. That is exact in exact arithmetic; in fp32 the `+ 1 - 1` rounds,
    to within 2.4e-7 of `nn.SELU` for |x| < 1 and 3 ulp beyond, far below the
    bf16 rounding Vollo computes in. Grouping the -1 with the exponential,
    `relu(x) / alpha + (min(exp(x), 1) - 1)`, avoids that but measured slower on
    every config, as did grouping it with the ReLU (SDK 29.0.0).

    Vollo fuses the clamp into the exp, the 1/alpha into the ReLU, and the outer
    `scale * alpha` and -1 into the next Linear, which leaves three pointwise
    instructions (ReLU, exp, add); a scale on the exp's output would cost a
    fourth (SDK 29.0.0).
    """

    _ALPHA = 1.6732632423543772848170429916717
    _SCALE = 1.0507009873554804934193349852946

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, sa = self._ALPHA, self._SCALE * self._ALPHA
        return sa * (torch.relu(x) / a + torch.clamp(torch.exp(x), max=1.0) - 1.0)

    def parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The two terms whose sum is the SELU, `scale * relu(x)` and
        `scale * alpha * (min(exp(x), 1) - 1)`, for a following Linear to sum
        in its accumulator (`ReadoutFNN.forward`)
        """
        a, sa = self._ALPHA, self._SCALE * self._ALPHA
        return torch.relu(x) * (sa / a), (torch.clamp(torch.exp(x), max=1.0) - 1.0) * sa


@beartype
def _round_robin_rows(rows: int, block_size: int, num_cores: int) -> list[int]:
    """
    Row counts of the first layer's pieces: two or more rounds of one piece per
    core, each piece at least two blocks (a Linear with one output block runs
    at half rate on the Agilex boards, SDK 29.0.0), and each core's total the
    compiler's own whole-block share of the layer. Returns `[rows]` (the layer
    left whole) on one core, or when the layer is too small for two rounds.
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


@beartype
def _placement(
    widths: list[int], block_size: int, num_cores: int, interleave: bool
) -> list[tuple[list[int], list[int]]]:
    """
    `(rows, cores)` of the pieces of the first Linear and, when it is a hidden
    layer too, of the second: each piece's output rows and the core it is
    pinned to, in output order. Empty on one core (every Linear left whole).

    Each core gets whole output blocks, as the compiler places a Linear, but
    the two layers' shares are chosen together: a core given an extra block of
    the first Linear gets no extra block of the second, which goes to the
    other end of the cores. Left to itself the compiler can give the cores
    that hold the first layer's extra blocks the second's extras too, and the
    second layer then waits on them (SDK 29.0.0).

    The first Linear is interleaved (`_round_robin_rows`, extras on the low
    cores) when `interleave` and it is large enough; otherwise it is one piece
    per core with its extra blocks on the high cores.
    """
    if num_cores == 1:
        return []
    rows = _round_robin_rows(widths[1], block_size, num_cores) if interleave else []
    if len(rows) > 1:
        cores = [i % num_cores for i in range(len(rows))]
        share = [0] * num_cores
        for r, c in zip(rows, cores):
            share[c] += -(-r // block_size)
        extras_low = True
    else:
        blocks = -(-widths[1] // block_size)
        rem = blocks % num_cores
        share = [blocks // num_cores + (c >= num_cores - rem) for c in range(num_cores)]
        rows, cores = _rows_per_core(widths[1], block_size, share)
        extras_low = False
    pieces = [(rows, cores)]
    if len(widths) > 3:
        blocks = -(-widths[2] // block_size)
        # Extra blocks to the cores holding the fewest of the first layer's,
        # from the end away from the first layer's extras
        order = sorted(
            range(num_cores), key=lambda c: (share[c], -c if extras_low else c)
        )
        second = [blocks // num_cores] * num_cores
        for c in order[: blocks % num_cores]:
            second[c] += 1
        pieces.append(_rows_per_core(widths[2], block_size, second))
    return pieces


@beartype
def _rows_per_core(
    rows: int, block_size: int, blocks: list[int]
) -> tuple[list[int], list[int]]:
    """One piece per core holding `blocks[c]` whole blocks, in core order"""
    out, cores, start = [], [], 0
    for c, n in enumerate(blocks):
        if n:
            end = min(rows, start + n * block_size)
            out.append(end - start)
            cores.append(c)
            start = end
    return out, cores


class _PinnedLinear(nn.Module):
    @beartype
    def __init__(self, in_features: int, rows: list[int], cores: list[int]):
        """
        `nn.Linear(in_features, sum(rows))` split by output rows into pieces of
        `rows` rows, piece `i` pinned to core `cores[i]`; the outputs are
        concatenated in order, so it computes exactly the plain Linear.

        The parameters live in the pieces (`pieces.<i>.weight`, as
        `named_parameters` shows), but the state dict presents them as a plain
        Linear's `weight` and `bias`, joined (a copy, not a view), and loads
        those back split across the pieces. So a plain `nn.Linear` checkpoint
        loads, and a missing or wrong-size one is reported as `weight`/`bias`.
        `.weight` and `.bias` also read the joined tensors, as copies.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = sum(rows)
        self.cores = cores
        self.pieces = nn.ModuleList(nn.Linear(in_features, r) for r in rows)

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([piece.weight for piece in self.pieces])

    @property
    def bias(self) -> torch.Tensor:
        return torch.cat([piece.bias for piece in self.pieces])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, in_features!]

        Returns:
            Tensor of shape [B, out_features!]
        """
        outputs = []
        for piece, core in zip(self.pieces, self.cores):
            with vollo_torch.CorePartition([core]):
                outputs.append(piece(x))  # [B, rows_i!]
        return torch.cat(outputs, dim=-1)  # [B, out_features!]

    def state_dict(self, *, destination=None, prefix="", keep_vars=False):
        """The pieces joined into a plain Linear's `weight` and `bias`"""
        if destination is None:
            destination = OrderedDict()
        for name in ("weight", "bias"):
            joined = torch.cat([getattr(piece, name) for piece in self.pieces])
            destination[prefix + name] = joined if keep_vars else joined.detach()
        return destination

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load a plain Linear's `weight` and `bias`, split into the pieces' keys
        for them to load. A missing or wrong-size tensor is reported under its
        plain name and leaves the pieces as they were.
        """
        names = {f"{prefix}weight", f"{prefix}bias"}
        for key in [k for k in state_dict if k.startswith(prefix) and k not in names]:
            unexpected_keys.append(key)
            del state_dict[key]
        rows = [piece.out_features for piece in self.pieces]
        shapes = {
            "weight": torch.Size([self.out_features, self.in_features]),
            "bias": torch.Size([self.out_features]),
        }
        for name, shape in shapes.items():
            key = prefix + name
            value = state_dict.pop(key, None)
            if value is None:
                missing_keys.append(key)
            elif value.shape != shape:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{value.shape} from checkpoint, the shape in current model "
                    f"is {shape}."
                )
            if value is None or value.shape != shape:
                parts = [getattr(piece, name).detach() for piece in self.pieces]
            else:
                parts = value.split(rows)
            for i, part in enumerate(parts):
                state_dict[f"{prefix}pieces.{i}.{name}"] = part


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
        interleave: bool = True,
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

        With `num_cores` > 1 the first two Linears are split by output rows
        into pieces pinned to the cores (`_placement`, `_PinnedLinear`), each
        output computed exactly as before:
        - With `interleave`, a large first Linear is split into two or more
          rounds of one piece per core (`_round_robin_rows`). Left whole, the
          compiler gives each core one contiguous run of the hidden units, and
          the second layer, which reads them in order, waits for each core's
          run to finish. Interleaved, the first round of pieces lands on every
          core early and the second layer starts on it while the cores
          compute the next round.
        - The second Linear's whole output blocks are shared out against the
          first's, so that a core holding an extra block of one holds no extra
          block of the other.
        - An output Linear of a single block is pinned to core 0.
        With SELU, the output Linear is applied to the last SELU's two terms
        and the results summed (`_SELU.parts`): the same function with the
        last add moved into the Linear's accumulator, where Vollo sums it
        instead of spending a pointwise instruction on it (SDK 29.0.0).
        The pinning uses `vollo_torch.CorePartition`, which is why this file
        imports `vollo_torch`; the network is otherwise plain PyTorch.

        `net` is the plain network's `nn.Sequential` (`net[0]` the first
        Linear, `net[2]` the second, ...) and the state dict is its layout
        whatever `num_cores` is, so a checkpoint of the plain network loads
        directly, and a checkpoint saved at one `num_cores` loads at any other.

        Args:
            n_inputs:   Trace length, I and Q interleaved (500 MS/s ADC: 2 ns each)
            hidden:     Hidden-layer widths
            n_outputs:  Joint states (2**qubits, or 3**qubits with leakage), or 1
            selu:       SELU as in Lienhard et al., or ReLU, which Vollo runs natively
            num_cores:  Cores to spread the first two Linears over (1: leave them whole)
            block_size: The config's block size, so pieces are whole blocks
            interleave: Interleave a large first Linear over the cores in rounds
        """
        super().__init__()
        widths = [n_inputs, *hidden, n_outputs]
        pieces = _placement(widths, block_size, num_cores, interleave)
        act = _SELU if selu else nn.ReLU
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(widths, widths[1:])):
            if i < len(pieces) and len(pieces[i][0]) > 1:
                layers.append(_PinnedLinear(d_in, *pieces[i]))
            else:
                layers.append(nn.Linear(d_in, d_out))
            layers.append(act())
        self.net = nn.Sequential(*layers[:-1])
        # An output Linear of one block measured faster on core 0 than where
        # the compiler puts it (SDK 29.0.0)
        self.output_core = 0 if num_cores > 1 and n_outputs <= block_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, n_inputs!]

        Returns:
            Tensor of shape [B, n_outputs!], logits (argmax on the controller)
        """
        *body, act, output = self.net
        for layer in body:
            x = layer(x)
        if isinstance(act, _SELU):
            # W (r + e) + b as W r + b + W e: the last SELU's add happens in
            # the output Linear's accumulator, not as a pointwise op
            x, e = act.parts(x)  # [B, hidden[-1]!] each
        else:
            x, e = act(x), None  # [B, hidden[-1]!]
        with (
            vollo_torch.CorePartition([self.output_core])
            if self.output_core is not None
            else contextlib.nullcontext()
        ):
            y = output(x)  # [B, n_outputs!]
            return y if e is None else y + F.linear(e, output.weight)


class ReadoutTeachers(nn.Module):
    @beartype
    def __init__(self, n_qubits: int, n_inputs: int, hidden: tuple[int, ...]):
        """
        KLiNQ's teachers (https://arxiv.org/abs/2503.03544): one `ReadoutFNN`
        per qubit, each reading the whole multiplexed trace and giving that
        qubit's logit, so each qubit is classified independently (mid-circuit
        measurement). KLiNQ deploys only small distilled students on its FPGA;
        Vollo runs the teachers themselves. Each teacher's first layer is left
        whole: `ReadoutTeachers` never interleaves.

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
        # Interleave the first layer only on the block-32 (six-core) boards,
        # where it measured faster (SDK 29.0.0); on the block-64 IA-840f it
        # measured slower on the SELU entries, so that board keeps it one
        # piece per core. HERQULES's and Mude's 500-wide layer is too small to
        # interleave anyway.
        hw = CONFIGS[config]
        interleave = hw.block_size == 32
        model = ReadoutFNN(
            n_inputs, hidden, n_outputs, selu, hw.num_cores, hw.block_size, interleave
        )
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
