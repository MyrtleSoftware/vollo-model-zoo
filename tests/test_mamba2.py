import numpy as np
import pytest
import torch
from beartype import beartype
from fla_mamba2 import Mamba2 as FLAMamba2

from vollo_model_zoo.models.mamba2 import Mamba2 as VolloMamba2


@beartype
def convert_state_dict(
    fla_state_dict: dict, n: int, d_state: int, n_heads: int, rmsnorm: bool
) -> dict:
    """
    Convert the state dict from an FLA Mamba2 block to a Vollo Mamba2 block.
    """
    vollo_state_dict = {}

    # in_proj split: [z, x, B, C, dt]
    weight = fla_state_dict["in_proj.weight"]

    vollo_state_dict["proj_z.weight"] = weight[0:n, :]
    vollo_state_dict["proj_x.weight"] = weight[n : 2 * n, :]
    vollo_state_dict["proj_B.weight"] = weight[2 * n : 2 * n + d_state, :]
    vollo_state_dict["proj_C.weight"] = weight[2 * n + d_state : 2 * n + 2 * d_state, :]
    vollo_state_dict["proj_dt.weight"] = weight[2 * n + 2 * d_state :, :]

    if "in_proj.bias" in fla_state_dict:
        bias = fla_state_dict["in_proj.bias"]
        vollo_state_dict["proj_z.bias"] = bias[0:n]
        vollo_state_dict["proj_x.bias"] = bias[n : 2 * n]
        vollo_state_dict["proj_B.bias"] = bias[2 * n : 2 * n + d_state]
        vollo_state_dict["proj_C.bias"] = bias[2 * n + d_state : 2 * n + 2 * d_state]
        vollo_state_dict["proj_dt.bias"] = bias[2 * n + 2 * d_state :]

    # conv1d split
    conv_weight = fla_state_dict["conv1d.weight"]
    vollo_state_dict["conv_x.conv1d.conv.weight"] = conv_weight[0:n, :, :]
    vollo_state_dict["conv_B.conv1d.conv.weight"] = conv_weight[n : n + d_state, :, :]
    vollo_state_dict["conv_C.conv1d.conv.weight"] = conv_weight[n + d_state :, :, :]

    if "conv1d.bias" in fla_state_dict:
        conv_bias = fla_state_dict["conv1d.bias"]
        vollo_state_dict["conv_x.conv1d.conv.bias"] = conv_bias[0:n]
        vollo_state_dict["conv_B.conv1d.conv.bias"] = conv_bias[n : n + d_state]
        vollo_state_dict["conv_C.conv1d.conv.bias"] = conv_bias[n + d_state :]

    # Simple mappings
    vollo_state_dict["dt_bias"] = fla_state_dict["dt_bias"]
    vollo_state_dict["A_log"] = fla_state_dict["A_log"]

    # D mapping: FLA D is [num_heads]
    D = fla_state_dict["D"]
    if D.shape[0] == n_heads:
        from einops import repeat

        vollo_state_dict["D"] = repeat(D, "h -> (h p)", p=n // n_heads)
    else:
        vollo_state_dict["D"] = D

    # Norm
    if rmsnorm and "norm.weight" in fla_state_dict:
        vollo_state_dict["norm.weight"] = fla_state_dict["norm.weight"]

    vollo_state_dict["out_proj.weight"] = fla_state_dict["out_proj.weight"]
    if "out_proj.bias" in fla_state_dict:
        vollo_state_dict["out_proj.bias"] = fla_state_dict["out_proj.bias"]

    return vollo_state_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("d_model", [32])
@pytest.mark.parametrize("d_state", [16])
@pytest.mark.parametrize("d_conv", [4])
@pytest.mark.parametrize("expand", [2])
@pytest.mark.parametrize("headdim", [8, 16])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("conv_bias", [True, False])
def test_mamba2_equivalence(
    d_model,
    d_state,
    d_conv,
    expand,
    headdim,
    bias,
    conv_bias,
):
    # Set seed
    torch.manual_seed(42)

    d_inner = int(expand * d_model)
    n_heads = d_inner // headdim

    fla_model = (
        FLAMamba2(
            num_heads=n_heads,
            head_dim=headdim,
            hidden_size=d_model,
            state_size=d_state,
            expand=expand,
            conv_kernel=d_conv,
            use_bias=bias,
            use_conv_bias=conv_bias,
            hidden_act="relu",
            rms_norm=True,
            backend="triton",
        )
        .eval()
        .cuda()
    )

    vollo_model = (
        VolloMamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=headdim,
            expand=expand,
            bias=bias,
            conv_bias=conv_bias,
            activation="relu",
        )
        .eval()
        .cuda()
    )

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(
        fla_model.state_dict(), d_inner, d_state, n_heads, True
    )
    vollo_model.load_state_dict(vollo_state_dict, strict=True)

    T = 8
    batch_size = 1

    x = torch.randn(batch_size, T, d_model).cuda()

    with torch.no_grad():
        fla_out = fla_model(x)

        vollo_out = vollo_model(x.squeeze(0))
        vollo_out = vollo_out.unsqueeze(0)

    # Check shapes
    assert fla_out.shape == vollo_out.shape

    # Check values
    np.testing.assert_allclose(
        fla_out.cpu().numpy(),
        vollo_out.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Mamba2 implementations output mismatch",
    )
