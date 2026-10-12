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
        channels = [8, 16, 32]
        convs, head, reference = _reference(stabilisers, rounds, channels)
        model = qec_tcn.FiLMDecoder(stabilisers, rounds, channels)
        model.load_reference(convs, head)

        events = torch.randint(0, 2, (1, 1, rounds, stabilisers)).float()
        torch.testing.assert_close(
            model(events.permute(0, 2, 3, 1))[0],
            reference(events),
            rtol=1e-4,
            atol=1e-4,
        )


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
    folded = qec_tcn.StreamingFiLMDecoder(stabilisers, channels, 3, 3, film_input=False)
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
