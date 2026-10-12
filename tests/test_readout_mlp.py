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
