import importlib

import pytest
import torch
from torch import nn

qec_tcn = importlib.import_module("vollo_model_zoo.models.qec-tcn")


def _reference(stabilisers, rounds, channels):
    """The paper's network: Conv2d blocks, flatten, dense head (FiLM as gamma, beta)."""
    ins = [1] + channels[:-1]
    convs = [nn.Conv2d(i, o, 3, padding=1) for i, o in zip(ins, channels)]
    head = nn.Linear(channels[-1] * rounds * stabilisers, stabilisers + 1)

    def forward(x, gamma=None, beta=None):  # x: [1, 1, rounds, stabilisers]
        for i, conv in enumerate(convs):
            x = conv(x)
            if gamma is not None:
                x = gamma[i][:, None, None] * x + beta[i][:, None, None]
            x = torch.relu(x)
        return head(x.flatten(1))

    return convs, head, forward


def test_window_decoder_matches_conv2d():
    torch.manual_seed(0)
    for stabilisers, rounds in [(4, 5), (6, 3), (10, 11)]:
        # The core placement (none, 3 or 6 cores) must not change the function.
        for num_cores in [None, 3, 6, [None, 3], [6, None]]:
            channels = [8, 16, 32]
            convs, head, reference = _reference(stabilisers, rounds, channels)
            model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels, num_cores)
            model.load_reference(convs, head)

            events = torch.randint(0, 2, (1, 1, rounds, stabilisers)).float()
            torch.testing.assert_close(
                model(events.permute(0, 2, 3, 1))[0],
                reference(events),
                rtol=1e-4,
                atol=1e-4,
            )


def test_window_decoder_has_no_padding_parameters():
    """The rewrite holds exactly the reference's weights, no zero padding."""
    stabilisers, rounds, channels = 10, 11, [128, 256, 512]
    convs, head, _ = _reference(stabilisers, rounds, channels)
    model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels)
    count = lambda modules: sum(p.numel() for m in modules for p in m.parameters())
    assert count([model]) == count(convs + [head])


def test_window_decoder_checkpoints_move_between_placements():
    """Each conv block saves and loads the `Conv2d`'s `weight` and `bias`."""
    torch.manual_seed(0)
    stabilisers, rounds, channels = 4, 5, [8, 16, 32]
    convs, head, reference = _reference(stabilisers, rounds, channels)
    source = qec_tcn.FiLMDecoder(stabilisers, rounds, channels)
    source.load_reference(convs, head)
    for i, ref in enumerate(convs[1:]):
        for name, value in ref.state_dict().items():
            torch.testing.assert_close(source.state_dict()[f"convs.{i}.{name}"], value)

    events = torch.randint(0, 2, (1, 1, rounds, stabilisers)).float()
    for num_cores in [None, 3, [None, 3], 6]:
        model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels, num_cores)
        model.load_state_dict(source.state_dict())
        assert list(model.state_dict()) == list(source.state_dict())
        torch.testing.assert_close(
            model(events.permute(0, 2, 3, 1))[0],
            reference(events),
            rtol=1e-4,
            atol=1e-4,
        )
        source = model  # and on to the next placement

    wider = qec_tcn.FiLMDecoder(stabilisers, rounds, [8, 16, 48])
    with pytest.raises(RuntimeError):
        source.load_state_dict(wider.state_dict())
    missing = {k: v for k, v in source.state_dict().items() if k != "convs.0.bias"}
    with pytest.raises(RuntimeError):
        source.load_state_dict(missing)
    extra = source.state_dict() | {"convs.0.rows.1.bias": torch.zeros(16)}
    with pytest.raises(RuntimeError):
        source.load_state_dict(extra)


def test_window_decoder_folded_film_matches_conv2d_film():
    torch.manual_seed(0)
    stabilisers, rounds, channels = 4, 5, [8, 16, 32]
    convs, head, reference = _reference(stabilisers, rounds, channels)
    gamma = [torch.rand(c) + 0.5 for c in channels]
    beta = [torch.randn(c) for c in channels]
    events = torch.randint(0, 2, (1, 1, rounds, stabilisers)).float()
    # Both lowerings of the conv blocks (unspread and spread over 6 cores).
    for num_cores in [None, 6]:
        model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels, num_cores)
        model.load_reference(convs, head)
        model.fold_film(gamma, beta)
        torch.testing.assert_close(
            model(events.permute(0, 2, 3, 1))[0],
            reference(events, gamma, beta),
            rtol=1e-4,
            atol=1e-4,
        )


def test_streaming_folded_film_matches_input_film():
    torch.manual_seed(0)
    stabilisers, channels, rounds = 6, [32, 64], 12
    dynamic = qec_tcn.StreamingFiLMDecoder(stabilisers, channels, 3, 3, film_input=True)
    # Pinning the first block to a core is placement only.
    folded = qec_tcn.StreamingFiLMDecoder(
        stabilisers, channels, 3, 3, film_input=False, first_core=5
    )
    folded.load_state_dict(dynamic.state_dict())

    gamma = [torch.rand(c) + 0.5 for c in channels]
    beta = [torch.randn(c) for c in channels]
    folded.fold_film(gamma, beta)

    syndrome = torch.randn(1, stabilisers, rounds)
    film = torch.cat([v for g, b in zip(gamma, beta) for v in (g, b)])
    film = film[None, :, None].expand(1, -1, rounds)
    torch.testing.assert_close(
        dynamic(torch.cat([syndrome, film], dim=1)), folded(syndrome)
    )


def test_window_decoder_spreads_kernel_rows_only_when_weight_copies_do_not_fit():
    channels = [128, 256, 512]
    # V80: 8 MiB of weight store per core holds a copy of every conv weight.
    assert qec_tcn._spread_cores("V80", 10, 11, channels) is None
    # IA-420f: 1 MiB per core does not, even with only the last block spread.
    assert qec_tcn._spread_cores("IA-420f", 4, 5, channels) == [6, 6]
    # IA-840f: the copies fit beside the d=5 head, not beside the d=11 head,
    # where spreading the last (largest) block is enough.
    assert qec_tcn._spread_cores("IA-840f", 4, 5, channels) is None
    assert qec_tcn._spread_cores("IA-840f", 10, 11, channels) == [None, 3]


def test_window_decoder_checks_arguments():
    with pytest.raises(ValueError):
        qec_tcn.FiLMDecoder(4, 1)
    with pytest.raises(ValueError):
        qec_tcn.FiLMDecoder(4, 5, num_cores=4)
    with pytest.raises(ValueError):
        qec_tcn.FiLMDecoder(4, 5, num_cores=[None, 4])
    with pytest.raises(ValueError):
        qec_tcn.FiLMDecoder(4, 5, num_cores=[None])


@pytest.mark.parametrize(
    "config, num_cores",
    [
        ("V80", None),
        ("V80", 6),
        ("IA-840f", None),
        ("IA-840f", 3),
        ("IA-840f", [None, 3]),
    ],
)
@pytest.mark.parametrize(
    "stabilisers, rounds, channels",
    # One block per layer, and several (the strided Conv1d over multiple blocks).
    [(4, 5, [8, 16, 32]), (4, 2, [8, 16, 32]), (4, 3, [64, 96, 160])],
)
def test_window_decoder_vm_matches_conv2d(
    config, num_cores, stabilisers, rounds, channels
):
    """The compiled lowering, for each placement `_vm` can pick, on the VM."""
    from vollo_model_zoo.vm import CONFIGS, _vollo_compile

    torch.manual_seed(0)
    convs, head, reference = _reference(stabilisers, rounds, channels)
    model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels, num_cores)
    model.load_reference(convs, head)
    events = (torch.rand(1, rounds, stabilisers, 1) < 0.3).float()
    program = _vollo_compile(
        model, events, time_axis=None, config=CONFIGS[config], quick_compile=True
    )
    out = program.to_vm(bit_accurate=True).run(events.numpy())
    with torch.no_grad():
        expected = reference(events.permute(0, 3, 1, 2))
    torch.testing.assert_close(
        torch.from_numpy(out).float().reshape(expected.shape),
        expected,
        rtol=0,
        atol=0.02 * float(expected.abs().max()) + 1e-3,
    )


def test_streaming_core_split_matches_unsplit():
    """Splitting the last block over cores, with FiLM input or folded, is exact."""
    torch.manual_seed(0)
    stabilisers, channels, rounds = 6, [32, 64], 12
    gamma = [torch.rand(c) + 0.5 for c in channels]
    beta = [torch.randn(c) for c in channels]
    syndrome = torch.randn(1, stabilisers, rounds)
    film = torch.cat([v for g, b in zip(gamma, beta) for v in (g, b)])
    film = film[None, :, None].expand(1, -1, rounds)

    plain = qec_tcn.StreamingFiLMDecoder(stabilisers, channels, 3, 3, film_input=True)
    for groups in [[32, 32], [16, 16, 16, 8, 8], [64]]:
        for film_input in [True, False]:
            split = qec_tcn.StreamingFiLMDecoder(
                stabilisers, channels, 3, 3, film_input, core_groups=groups
            )
            split.convs[0].load_state_dict(plain.convs[0].state_dict())
            split.convs[1].load_conv(plain.convs[1])
            split.head.load_state_dict(plain.head.state_dict())
            if film_input:
                out = split(torch.cat([syndrome, film], dim=1))
            else:
                split.fold_film(gamma, beta)
                out = split(syndrome)
            torch.testing.assert_close(out, plain(torch.cat([syndrome, film], dim=1)))


def test_streaming_core_split_checkpoints_move_between_groupings():
    """The split block saves and loads the unsplit `convs.1.conv.*` layout."""
    torch.manual_seed(0)
    stabilisers, channels = 6, [32, 64]
    plain = qec_tcn.StreamingFiLMDecoder(stabilisers, channels, 3, 3, film_input=True)
    six = qec_tcn.StreamingFiLMDecoder(
        stabilisers, channels, 3, 3, film_input=True, core_groups=[16, 16, 8, 8, 8, 8]
    )
    three = qec_tcn.StreamingFiLMDecoder(
        stabilisers, channels, 3, 3, film_input=True, core_groups=[32, 32]
    )
    six.load_state_dict(plain.state_dict())
    assert list(six.state_dict()) == list(plain.state_dict())
    three.load_state_dict(six.state_dict())
    x = torch.randn(1, stabilisers + 2 * sum(channels), 7)
    for model in (six, three):
        torch.testing.assert_close(model(x), plain(x))

    wider = qec_tcn.StreamingFiLMDecoder(stabilisers, [32, 96], 3, 3)
    with pytest.raises(ValueError):
        six.convs[1].load_conv(wider.convs[1])
    with pytest.raises(RuntimeError):
        six.load_state_dict(wider.state_dict())
