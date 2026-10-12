"""
The SELU rebuilt from supported ops against `torch.nn.SELU`, which the Vollo
compiler does not accept.
"""

import importlib

import torch

_SELU = importlib.import_module("vollo_model_zoo.models.readout-mlp")._SELU


def test_selu_matches_torch():
    x = torch.cat([torch.linspace(-20, 20, 4001), torch.randn(1000) * 3])
    torch.testing.assert_close(_SELU()(x), torch.nn.SELU()(x), atol=1e-5, rtol=1e-5)


def test_features_in_students_match_full_bank():
    m = importlib.import_module("vollo_model_zoo.models.readout-mlp")
    f = (31, 201, 201, 31, 31)
    full = m.QubitReadoutStudents(500, f).eval()
    bare = m.QubitReadoutStudents(500, f, front_end=False).eval()
    for s, t in zip(full.students, bare.students):
        t.net.load_state_dict(s.net.state_dict())
    x = torch.randn(2, 5, 1000)
    feats = torch.cat([s.front_end(x[:, q]) for q, s in enumerate(full.students)], -1)
    torch.testing.assert_close(bare(feats), full(x))
