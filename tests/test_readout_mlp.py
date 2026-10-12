"""
The SELU rebuilt from supported ops against `torch.nn.SELU`, which the Vollo
compiler does not accept, and the interleaved first layer against the plain
network, including loading its checkpoints.
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
    """The paper's network, in the layout of a plain checkpoint: `net` is an
    `nn.Sequential` of Linears with nn.SELU or nn.ReLU between them."""
    layers = []
    for d_in, d_out in zip(widths, widths[1:]):
        linear = torch.nn.Linear(d_in, d_out)
        torch.nn.init.normal_(linear.weight, std=d_in**-0.5)
        torch.nn.init.normal_(linear.bias, std=0.1)
        layers += [linear, torch.nn.SELU() if selu else torch.nn.ReLU()]
    reference = torch.nn.Module()
    reference.net = torch.nn.Sequential(*layers[:-1])
    reference.forward = reference.net.forward
    return reference


_WIDTHS = [
    ((1000, 1000, 500, 250, 32), True),
    ((800, 800, 400, 200, 32), True),
    ((1000, 500, 250, 243), False),
]
_CORES = [(1, 32), (6, 32), (3, 64)]


def _model(widths, selu, num_cores, block_size):
    return readout_mlp.ReadoutFNN(
        widths[0], widths[1:-1], widths[-1], selu, num_cores, block_size
    )


@pytest.mark.parametrize("num_cores, block_size", _CORES)
@pytest.mark.parametrize("widths, selu", _WIDTHS)
def test_plain_checkpoint_loads(widths, selu, num_cores, block_size):
    torch.manual_seed(0)
    reference = _plain(widths, selu)
    model = _model(widths, selu, num_cores, block_size)
    model.load_state_dict(reference.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )

    x = torch.randn(64, widths[0])
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("widths, selu", _WIDTHS)
def test_checkpoint_round_trips(widths, selu):
    torch.manual_seed(0)
    reference = _plain(widths, selu)
    x = torch.randn(64, widths[0])
    with torch.no_grad():
        expected = reference(x)

    saved = _model(widths, selu, 6, 32)
    saved.load_state_dict(reference.state_dict())
    state = saved.state_dict()
    # Saved in the plain layout, whatever the split
    assert state.keys() == reference.state_dict().keys()

    for num_cores, block_size in _CORES:
        model = _model(widths, selu, num_cores, block_size)
        model.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected, atol=1e-5, rtol=1e-5)
    plain = _plain(widths, selu)
    plain.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(plain(x), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_cores, block_size", _CORES)
@pytest.mark.parametrize(
    "widths, bad",
    [
        # Only the first layer is wrong: the trace length, or one row short
        ((1000, 1000, 500, 250, 32), (900, 1000, 500, 250, 32)),
        ((1000, 1000, 500, 250, 32), "row"),
        # The first two layers are wrong
        ((1000, 1000, 500, 250, 32), (1000, 960, 500, 250, 32)),
    ],
)
def test_wrong_size_first_layer_is_reported(widths, bad, num_cores, block_size):
    state = _plain(widths if bad == "row" else bad, True).state_dict()
    if bad == "row":
        state["net.0.weight"] = state["net.0.weight"][:-1]
    model = _model(widths, True, num_cores, block_size)
    before = model.state_dict()
    with pytest.raises(RuntimeError, match=r"size mismatch for net\.0\.weight") as e:
        model.load_state_dict(state)
    # The error names the plain checkpoint's keys, never the pieces'
    assert "pieces" not in str(e.value)
    # Not strict still raises on a size mismatch, and leaves the first layer as it was
    with pytest.raises(RuntimeError, match=r"size mismatch for net\.0\.weight"):
        model.load_state_dict(state, strict=False)
    torch.testing.assert_close(
        model.state_dict()["net.0.weight"], before["net.0.weight"]
    )


@pytest.mark.parametrize("num_cores, block_size", _CORES)
def test_missing_first_layer_is_reported(num_cores, block_size):
    widths = (1000, 1000, 500, 250, 32)
    state = _plain(widths, True).state_dict()
    del state["net.0.weight"], state["net.0.bias"]
    model = _model(widths, True, num_cores, block_size)
    with pytest.raises(RuntimeError) as e:
        model.load_state_dict(state)
    assert "net.0.weight" in str(e.value) and "net.0.bias" in str(e.value)
    assert "pieces" not in str(e.value)
    result = model.load_state_dict(state, strict=False)
    assert result.missing_keys == ["net.0.weight", "net.0.bias"]
    assert result.unexpected_keys == []


@pytest.mark.parametrize("num_cores, block_size", _CORES)
def test_layout_matches_plain(num_cores, block_size):
    """`net` indexes, names and orders like the plain network, at any split"""
    widths = (1000, 1000, 500, 250, 32)
    reference = _plain(widths, True)
    model = _model(widths, True, num_cores, block_size)
    model.load_state_dict(reference.state_dict())
    assert list(model.state_dict()) == list(reference.state_dict())
    assert len(model.net) == len(reference.net)
    for ours, plain in zip(model.net, reference.net):
        if isinstance(plain, torch.nn.Linear):
            torch.testing.assert_close(ours.weight, plain.weight, rtol=0, atol=0)
            torch.testing.assert_close(ours.bias, plain.bias, rtol=0, atol=0)
            assert (ours.in_features, ours.out_features) == (
                plain.in_features,
                plain.out_features,
            )
        else:
            assert not isinstance(ours, torch.nn.Linear)


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


@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("num_cores, block_size", _CORES)
@pytest.mark.parametrize(
    "widths",
    [
        (1000, 1000, 500, 250, 32),
        (800, 800, 400, 200, 32),
        (1000, 500, 250, 32),
        (1000, 500, 250, 243),
        (100, 64, 32),
    ],
)
def test_placement(widths, num_cores, block_size, interleave):
    pieces = readout_mlp._placement(list(widths), block_size, num_cores, interleave)
    if num_cores == 1:
        assert pieces == []
        return
    # The first Linear, and the second when it is a hidden layer
    assert len(pieces) == (2 if len(widths) > 3 else 1)
    shares = []
    for (rows, cores), width in zip(pieces, widths[1:]):
        assert sum(rows) == width and len(rows) == len(cores)
        assert set(cores) <= set(range(num_cores))
        share = [0] * num_cores
        for r, c in zip(rows, cores):
            share[c] += -(-r // block_size)
        # Each core holds whole blocks: the compiler's own share of the layer
        total = -(-width // block_size)
        assert sum(share) == total
        assert max(share) == -(-total // num_cores)
        assert min(share) == total // num_cores
        shares.append(share)
    if len(shares) == 2:
        # No core holds an extra block of both layers while another holds
        # extras of neither
        first, second = shares
        extra = [f > min(first) for f in first], [s > min(second) for s in second]
        both = any(a and b for a, b in zip(*extra))
        neither = any(not a and not b for a, b in zip(*extra))
        assert not (both and neither)


def test_placement_at_six_cores():
    """The measured placements (SDK 29.0.0): second layer's extras away from the first's"""
    lienhard = readout_mlp._placement([1000, 1000, 500, 250, 32], 32, 6, True)
    assert lienhard[0][1] == [i % 6 for i in range(12)]
    assert lienhard[1] == ([64, 64, 96, 96, 96, 84], [0, 1, 2, 3, 4, 5])
    herq = readout_mlp._placement([1000, 500, 250, 32], 32, 6, True)
    assert herq[0] == ([64, 64, 96, 96, 96, 84], [0, 1, 2, 3, 4, 5])
    assert herq[1] == ([64, 64, 32, 32, 32, 26], [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("widths, selu", _WIDTHS)
def test_one_piece_per_core_loads(widths, selu):
    """The IA-840f layout: no rounds, both Linears one piece per core"""
    torch.manual_seed(0)
    reference = _plain(widths, selu)
    model = readout_mlp.ReadoutFNN(
        widths[0], widths[1:-1], widths[-1], selu, 3, 64, interleave=False
    )
    assert isinstance(model.net[0], readout_mlp._PinnedLinear)
    assert isinstance(model.net[2], readout_mlp._PinnedLinear)
    model.load_state_dict(reference.state_dict())
    assert list(model.state_dict()) == list(reference.state_dict())
    x = torch.randn(64, widths[0])
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(x), atol=1e-5, rtol=1e-5)


def test_output_core():
    """A single-block output Linear is pinned to core 0, a wider one left to the compiler"""
    assert readout_mlp.ReadoutFNN(1000, (500, 250), 32, False, 6, 32).output_core == 0
    assert (
        readout_mlp.ReadoutFNN(1000, (500, 250), 243, False, 6, 32).output_core is None
    )
    assert readout_mlp.ReadoutFNN(1000, (500, 250), 32, False).output_core is None
