import importlib
import math

import pytest
import torch

neural_bp = importlib.import_module("vollo_model_zoo.models.neural-bp")


def toric_checks(L: int) -> list[list[int]]:
    """Qubits of each plaquette: A(i, j) = i*L + j, B(i, j) = L*L + i*L + j."""
    return [
        [
            i * L + j,
            i * L + (j + 1) % L,
            L * L + i * L + j,
            L * L + (i + 1) % L * L + j,
        ]
        for i in range(L)
        for j in range(L)
    ]


def reference_bp(checks, n, syndrome, p, rule, w, b, eps=neural_bp._EPS):
    """Textbook neural BP (Eqs. 4-6) with every message in a dict keyed by
    (check, qubit). w[t] is [4, m] over (slot, check), b[t] is [2, m] over qubits."""
    prior = math.log((1 - p) / p)
    edges = [(c, q) for c, qs in enumerate(checks) for q in qs]
    slot = {(c, q): k for c, qs in enumerate(checks) for k, q in enumerate(qs)}
    to_check = {(c, q): prior * float(b[0].reshape(-1)[q]) for c, q in edges}
    for t in range(len(w)):
        to_var = {}
        for c, q in edges:
            others = [to_check[c, r] for r in checks[c] if r != q]
            if rule == "sum-product":
                prod = math.prod(math.tanh(x / 2) for x in others)
                prod = max(-1 + eps, min(1 - eps, prod))
                msg = 2 * math.atanh(prod)
            else:
                signs = math.prod(math.copysign(1, x) if x else 0 for x in others)
                msg = signs * min(abs(x) for x in others)
            to_var[c, q] = (-1) ** int(syndrome[c]) * msg * float(w[t][slot[c, q], c])
        bias = [prior * float(b[t + 1].reshape(-1)[q]) for q in range(n)]
        if t == len(w) - 1:
            return [
                bias[q] + sum(to_var[e] for e in edges if e[1] == q) for q in range(n)
            ]
        to_check = {
            (c, q): bias[q] + sum(to_var[e] for e in edges if e[1] == q and e[0] != c)
            for c, q in edges
        }


@pytest.mark.parametrize("rule", ["sum-product", "min-sum"])
@pytest.mark.parametrize("random_weights", [False, True])
def test_matches_reference_bp(rule, random_weights):
    torch.manual_seed(0)
    L, iterations, p = 4, 3, 0.05
    if rule == "min-sum":
        # Integer LLRs keep every message exact, so signs never near-cancel.
        p = 1 / (1 + math.exp(3))
    model = neural_bp.NeuralBP(L, iterations, rule, p)
    if random_weights:
        for param in [*model.prior_weights, *model.check_weights]:
            param.data = 0.5 + torch.rand_like(param)
    syndromes = (torch.rand(8, L * L) < 0.2).float()
    out = model(syndromes).reshape(8, -1)
    checks = toric_checks(L)
    for s, o in zip(syndromes, out):
        expected = reference_bp(
            checks,
            2 * L * L,
            s,
            p,
            rule,
            list(model.check_weights),
            list(model.prior_weights),
        )
        torch.testing.assert_close(
            o, torch.tensor(expected, dtype=o.dtype), atol=1e-3, rtol=1e-3
        )
