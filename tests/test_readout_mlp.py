"""
The SELU rebuilt from supported ops against `torch.nn.SELU`, which the Vollo
compiler does not accept.
"""

import copy
import importlib

import torch

readout_mlp = importlib.import_module("vollo_model_zoo.models.readout-mlp")
_SELU = readout_mlp._SELU


def test_selu_matches_torch():
    # exp overflows to inf beyond ~88 and the clamp must still give exactly 1
    extremes = torch.tensor([-1e4, -100.0, -88.0, 88.0, 100.0, 1e4])
    x = torch.cat([torch.linspace(-20, 20, 4001), torch.randn(1000) * 3, extremes])
    torch.testing.assert_close(_SELU()(x), torch.nn.SELU()(x), atol=1e-5, rtol=1e-5)


def test_selu_network_matches_torch():
    torch.manual_seed(0)
    model = readout_mlp.ReadoutFNN(1000, (1000, 500, 250), 32, selu=True)
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=layer.in_features**-0.5)
            torch.nn.init.normal_(layer.bias, std=0.1)
    reference = copy.deepcopy(model)
    for i, layer in enumerate(reference.net):
        if isinstance(layer, _SELU):
            reference.net[i] = torch.nn.SELU()

    x = torch.randn(64, 1000)
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(x), atol=1e-5, rtol=1e-5)
