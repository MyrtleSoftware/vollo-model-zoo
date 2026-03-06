import numpy as np
import pytest
import torch
from torch import nn

from vollo_model_zoo.models.gru import GRU as VolloGRU


def convert_state_dict(torch_gru: nn.GRU, vollo_gru: VolloGRU):
    """
    Convert the state dict from a torch.nn.GRU to a Vollo GRU.
    """
    torch_state_dict = torch_gru.state_dict()
    vollo_state_dict = {}

    n = torch_gru.num_layers
    h = torch_gru.hidden_size

    for i in range(n):
        # PyTorch weights are (3*hidden_size, input_size) or (3*hidden_size, hidden_size)
        # and are concatenated in the order (r, z, n)

        weight_ih = torch_state_dict[f"weight_ih_l{i}"]
        weight_hh = torch_state_dict[f"weight_hh_l{i}"]

        # Vollo GRU layers are in a Sequential called 'layers'
        # Each layer has a 'scan' module with a 'step' module
        prefix = f"layers.{i}.scan.step"

        vollo_state_dict[f"{prefix}.linear_ih_r.weight"] = weight_ih[0:h]
        vollo_state_dict[f"{prefix}.linear_ih_z.weight"] = weight_ih[h : 2 * h]
        vollo_state_dict[f"{prefix}.linear_ih_n.weight"] = weight_ih[2 * h : 3 * h]

        vollo_state_dict[f"{prefix}.linear_hh_r.weight"] = weight_hh[0:h]
        vollo_state_dict[f"{prefix}.linear_hh_z.weight"] = weight_hh[h : 2 * h]
        vollo_state_dict[f"{prefix}.linear_hh_n.weight"] = weight_hh[2 * h : 3 * h]

        if torch_gru.bias:
            bias_ih = torch_state_dict[f"bias_ih_l{i}"]
            bias_hh = torch_state_dict[f"bias_hh_l{i}"]

            vollo_state_dict[f"{prefix}.linear_ih_r.bias"] = bias_ih[0:h]
            vollo_state_dict[f"{prefix}.linear_ih_z.bias"] = bias_ih[h : 2 * h]
            vollo_state_dict[f"{prefix}.linear_ih_n.bias"] = bias_ih[2 * h : 3 * h]

            vollo_state_dict[f"{prefix}.linear_hh_r.bias"] = bias_hh[0:h]
            vollo_state_dict[f"{prefix}.linear_hh_z.bias"] = bias_hh[h : 2 * h]
            vollo_state_dict[f"{prefix}.linear_hh_n.bias"] = bias_hh[2 * h : 3 * h]

    return vollo_state_dict


@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", [16, 32])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("fp32", [True, False])
def test_gru_equivalence(input_size, hidden_size, num_layers, bias, fp32):
    # Set seed
    torch.manual_seed(42)

    torch_model = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=False,  # Vollo GRU expects [T, input_size]
    ).eval()

    vollo_model = VolloGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        fp32=fp32,
    ).eval()

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(torch_model, vollo_model)
    vollo_model.load_state_dict(vollo_state_dict, strict=True)

    T = 10
    # torch_model expects [T, B, input_size] when batch_first=False
    x_torch = torch.randn(T, 1, input_size)
    # vollo_model expects [T, input_size]
    x_vollo = x_torch.squeeze(1)

    with torch.no_grad():
        torch_out, _ = torch_model(x_torch)
        vollo_out = vollo_model(x_vollo)

    # torch_out is [T, 1, hidden_size]
    torch_out = torch_out.squeeze(1)

    # Check shapes
    assert torch_out.shape == vollo_out.shape

    np.testing.assert_allclose(
        torch_out.numpy(),
        vollo_out.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="GRU implementations output mismatch",
    )
