import importlib

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
        for num_cores in [None, 3, 6]:
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


def test_window_decoder_folded_film_matches_conv2d_film():
    torch.manual_seed(0)
    stabilisers, rounds, channels = 4, 5, [8, 16, 32]
    convs, head, reference = _reference(stabilisers, rounds, channels)
    model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels)
    model.load_reference(convs, head)
    gamma = [torch.rand(c) + 0.5 for c in channels]
    beta = [torch.randn(c) for c in channels]
    model.fold_film(gamma, beta)

    events = torch.randint(0, 2, (1, 1, rounds, stabilisers)).float()
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


def test_window_decoder_spreads_kernel_rows_only_when_weights_cannot_be_copied():
    channels = [128, 256, 512]
    # V80: 8 MiB of weight store per core holds a copy of every conv weight.
    assert qec_tcn._spread_cores("V80", 10, 11, channels) is None
    # IA-420f: 1 MiB per core does not.
    assert qec_tcn._spread_cores("IA-420f", 4, 5, channels) == 6
    # IA-840f: the copies fit beside the d=5 head, not beside the d=11 head.
    assert qec_tcn._spread_cores("IA-840f", 4, 5, channels) is None
    assert qec_tcn._spread_cores("IA-840f", 10, 11, channels) == 3


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
