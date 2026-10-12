"""
The SELU rebuilt from supported ops against `torch.nn.SELU`, which the Vollo
compiler does not accept, and the interleaved first layer against the plain
network.
"""

import importlib

import pytest
import torch

readout_mlp = importlib.import_module("vollo_model_zoo.models.readout-mlp")
_SELU = readout_mlp._SELU


def test_selu_matches_torch():
    # exp overflows to inf beyond ~88 and the clamp must still give exactly 1
    extremes = torch.tensor([-1e4, -100.0, -88.0, 88.0, 100.0, 1e4])
    x = torch.cat([torch.linspace(-20, 20, 4001), torch.randn(1000) * 3, extremes])
    torch.testing.assert_close(_SELU()(x), torch.nn.SELU()(x), atol=1e-5, rtol=1e-5)


def _plain(widths, selu):
    """The paper's network: plain Linears, nn.SELU or nn.ReLU between them."""
    layers = []
    for d_in, d_out in zip(widths, widths[1:]):
        linear = torch.nn.Linear(d_in, d_out)
        torch.nn.init.normal_(linear.weight, std=d_in**-0.5)
        torch.nn.init.normal_(linear.bias, std=0.1)
        layers += [linear, torch.nn.SELU() if selu else torch.nn.ReLU()]
    return torch.nn.Sequential(*layers[:-1])


@pytest.mark.parametrize("num_cores, block_size", [(1, 32), (6, 32), (3, 64)])
@pytest.mark.parametrize(
    "widths, selu",
    [
        ((1000, 1000, 500, 250, 32), True),
        ((800, 800, 400, 200, 32), True),
        ((1000, 500, 250, 243), False),
    ],
)
def test_network_matches_plain(widths, selu, num_cores, block_size):
    torch.manual_seed(0)
    reference = _plain(widths, selu)
    model = readout_mlp.ReadoutFNN(
        widths[0], widths[1:-1], widths[-1], selu, num_cores, block_size
    )
    model.load_reference([m for m in reference if isinstance(m, torch.nn.Linear)])
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )

    x = torch.randn(64, widths[0])
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("rows", [200, 250, 400, 500, 800, 1000, 1024, 2000])
@pytest.mark.parametrize("num_cores, block_size", [(1, 32), (6, 32), (3, 64)])
def test_round_robin_rows(rows, num_cores, block_size):
    pieces = readout_mlp._round_robin_rows(rows, block_size, num_cores)
    assert sum(pieces) == rows
    if num_cores == 1 or len(pieces) == 1:
        assert pieces == [rows]
        return
    blocks = [-(-p // block_size) for p in pieces]
    assert len(pieces) % num_cores == 0 and len(pieces) >= 2 * num_cores
    assert min(blocks) >= 2
    # Each core's total is its whole-block share of the layer, as the compiler
    # would place it unsplit
    per_core = sorted(sum(blocks[c::num_cores]) for c in range(num_cores))
    total = -(-rows // block_size)
    assert per_core[-1] == -(-total // num_cores)
    assert per_core[0] == total // num_cores
