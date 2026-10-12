import math
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import torch
from beartype import beartype
from torch import nn

_EPS = 2**-6  # keeps atanh finite in bf16


def _shift(L: int, di: int, dj: int) -> torch.Tensor:
    """Permutation P with (P @ x)[i, j] = x[i + di, j + dj] on an L x L torus."""
    P = torch.zeros(L * L, L * L)
    for i in range(L):
        for j in range(L):
            P[i * L + j, (i + di) % L * L + (j + dj) % L] = 1
    return P


def _others(x: torch.Tensor, op) -> torch.Tensor:
    """For each of the 4 slots, `op` reduced over the other three, in a tree of depth 2."""
    a, b, c = [torch.cat([x[:, i:], x[:, :i]], dim=1) for i in (1, 2, 3)]
    return op(op(a, b), c)


class NeuralBP(nn.Module):
    @beartype
    def __init__(
        self,
        side: int,
        iterations: int,
        check_update: Literal["sum-product", "min-sum"],
        p: float = 0.01,
    ):
        """
        Neural belief-propagation decoder for the toric code, after Liu & Poulin
        (https://arxiv.org/abs/1811.07835).

        Quantum mapping: a round of plaquette (Z-check) measurements on the L x L
        toric code gives a syndrome; belief propagation on its Tanner graph,
        unrolled for `iterations` check-node passes, estimates the posterior
        log-likelihood ratio of an X error on each of the 2 L^2 qubits. It is the
        fast first-stage decoder: confident qubits are corrected directly, the
        rest go to a slower one. The paper decodes X and Z errors separately, so
        this X-only model is the whole of one decoder.

        The weights are those of the paper's Eqs. 4-6, with mu_cv the
        check-to-variable messages and l_v the prior LLR:

            mu_vc = l_v b_v + sum_{c' != c} w_{c'v,vc} mu_c'v        (Eq. 4)
            mu_v  = l_v b_v + sum_c w_{cv,v} mu_cv                   (Eq. 6)

        `b` is one bias per qubit per iteration, [iterations + 1] tensors of shape
        [2, m] for qubits A and B, and `w` one weight per check-to-variable edge
        per iteration, [iterations] tensors of shape [4, m]. A qubit has degree 2,
        so the weight on the pair (incoming edge c'v, outgoing edge vc) is fixed
        by the incoming edge alone: one weight per incoming edge is the full set.
        Both start at 1, which is plain BP.

        Not implemented: the paper's residual connections between cycles (given
        only in its supplementary material) and its weight sharing under lattice
        translation, so weights do not transfer between L and trained weights
        from elsewhere need mapping into the shapes above.

        A plaquette (i, j) has four edges, slots 0..3, to qubits A(i, j),
        A(i, j+1), B(i, j) and B(i+1, j). Messages are [batch, slot, plaquette].
        A qubit sits in two plaquettes, so its variable node update is just the
        other plaquette's message: a constant shift of the plaquette grid between
        slots 0 <-> 1 and 2 <-> 3.

        Args:
            side:         Torus side L: m = L^2 checks, n = 2 L^2 qubits
            iterations:   Number of unrolled check-node passes T (the paper uses
                          25 for the toric code)
            check_update: Check node rule, "sum-product" (tanh/atanh, the
                          paper's rule) or "min-sum" (sign product times the
                          minimum magnitude; a faster variant, not the paper's)
            p:            Physical error rate, setting the prior LLR log((1-p)/p)
                          (the paper uses 0.01 for the toric code)
        """
        super().__init__()
        self.m, self.check_update = side * side, check_update
        self.prior = math.log((1 - p) / p)
        # Constant 0/1 matrices: they count towards param_count but are not trained.
        self.shifts = nn.ModuleList()
        for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            shift = nn.Linear(self.m, self.m, bias=False)
            shift.weight = nn.Parameter(_shift(side, di, dj), requires_grad=False)
            self.shifts.append(shift)
        # b^(0..T) per qubit [2, m]; w^(1..T) per check-to-variable edge [4, m].
        self.prior_weights = nn.ParameterList(
            nn.Parameter(torch.ones(2, self.m)) for _ in range(iterations + 1)
        )
        self.check_weights = nn.ParameterList(
            nn.Parameter(torch.ones(4, self.m)) for _ in range(iterations)
        )

    def _check_node(self, mu: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Check-to-variable messages from variable-to-check messages mu.

        `weight` carries the syndrome and the edge weight: a violated check flips
        the sign of everything it sends.
        """
        if self.check_update == "sum-product":
            a = torch.tanh(mu / 2)
            p = _others(a, torch.mul).clamp(-1 + _EPS, 1 - _EPS)
            return (torch.log(1 + p) - torch.log(1 - p)) * weight  # 2 atanh(p)
        # sign(mu) as a steep clamp, since there is no sign op; |mu| = mu * sign(mu).
        sign = (mu * 256).clamp(-1, 1)
        mag = mu * sign
        return _others(mag, torch.minimum) * (_others(sign, torch.mul) * weight)

    def _variable_node(
        self, mu_cv: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Variable-to-check messages mu_vc = l_v b_v + (other edge's message), and
        the unshifted sum u = l_v b_v + (other edge's message) for slots 0 and 2.

        Each edge receives the message of the other edge of its qubit, shifted
        from the neighbouring plaquette. Slots 1 and 3 hold the qubits of slots 0
        and 2 seen from the next plaquette, so their prior is the same b shifted:
        s1(c0) + s1(l b_A) = s1(c0 + l b_A). Adding l b before the shift keeps
        every add a [2, m] constant against a runtime tensor, which the compiler
        folds (a runtime-built [4, m] prior added each iteration costs ~25% more
        per iteration with min-sum).
        """
        c0, c1, c2, c3 = [mu_cv[:, k : k + 1] for k in range(4)]
        s0, s1, s2, s3 = self.shifts
        lb = self.prior * b  # [2, m!]
        u = torch.cat([s0(c1), s2(c3)], dim=1) + lb  # [batch, 2, m!] slots 0, 2
        v = torch.cat([c0, c2], dim=1) + lb  # [batch, 2, m!] before the shift
        mu = torch.cat([u[:, 0:1], s1(v[:, 0:1]), u[:, 1:2], s3(v[:, 1:2])], dim=1)
        return mu, u

    def _initial_prior(self, b: torch.Tensor, one: torch.Tensor) -> torch.Tensor:
        """l_v * b_v onto the edges [batch, 4, m]: slots 1 and 3 hold qubits
        A(i, j+1) and B(i+1, j), shifted copies of slots 0, 2. Runs once.
        `one` is an all-ones [batch, 1, m] input-dependent tensor: the compiler
        does not constant-fold, so slicing or shifting b alone is rejected."""
        _, s1, _, s3 = self.shifts
        b = self.prior * b * one  # [batch, 2, m!]
        a, c = b[:, 0:1], b[:, 1:2]
        return torch.cat([a, s1(a), c, s3(c)], dim=1)

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Args:
            syndrome: Tensor of shape (Batch, m) with entries in {0, 1}

        Returns:
            Tensor of shape (Batch, 2, m): posterior error LLR of qubits A and B
        """
        sign = (1 - 2 * syndrome).unsqueeze(1)  # [batch, 1, m!]
        # == 1: a constant-only start cannot be sliced by the compiler
        one = sign * sign
        mu = self._initial_prior(self.prior_weights[0], one)
        for w, b in zip(self.check_weights[:-1], self.prior_weights[1:-1]):
            mu, _ = self._variable_node(self._check_node(mu, w * sign), b)
        mu_cv = self._check_node(mu, self.check_weights[-1] * sign)
        # Eq. 6: slots 0 and 2 are the two qubits; u holds their prior and the
        # message from their other plaquette, mu_cv their own plaquette's.
        _, u = self._variable_node(mu_cv, self.prior_weights[-1])
        return u + torch.cat([mu_cv[:, 0:1], mu_cv[:, 2:3]], dim=1)


@beartype
def _vm(
    side: int,
    iterations: int,
    check_update: Literal["sum-product", "min-sum"],
    config: str,
):
    from vollo_model_zoo.vm import vollo_info

    syndrome = (torch.rand(1, side * side) < 0.2).float()
    model = NeuralBP(side, iterations, check_update)
    return vollo_info(
        model,
        syndrome,
        config=config,
        time_axis=None,
        meta=dict(L=side, T=iterations, check_update=check_update),
    )


@beartype
def main(config: str = "V80") -> Generator:
    for x in [
        dict(side=4, iterations=5),
        # iterations=25 is the paper's N_c for the toric code
        dict(side=4, iterations=25),
        dict(side=6, iterations=5),
        dict(side=6, iterations=25),
        dict(side=8, iterations=5),
        dict(side=8, iterations=10),
        dict(side=8, iterations=25),
        dict(side=12, iterations=5),
        dict(side=12, iterations=10),
        # ~1M parameters baseline: the four m x m shift matrices dominate,
        # 4 m^2 = 4 * 22^4 = 937k, plus b: 2 m (T + 1) = 2 * 484 * 11 = 10.6k
        # and w: 4 m T = 4 * 484 * 10 = 19.4k, total 967k.
        # Far above a decoding round (~1 us); the smaller sizes are the ones
        # that fit a decoder's time budget.
        dict(side=22, iterations=10),
    ]:
        # Sum-product is the paper's rule; min-sum is a faster variant (not the
        # paper's) that drops tanh/atanh for a shorter chain.
        for check_update in ["sum-product", "min-sum"]:
            yield _vm(**x, check_update=check_update, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
