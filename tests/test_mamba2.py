import numpy as np
import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from mamba2_ref import Mamba2 as Mamba2Ref

from vollo_model_zoo.models.mamba2 import Mamba2 as Mamba2Vollo


@beartype
def convert_state_dict(
    ref_state_dict: dict, d_inner: int, d_state: int, n_heads: int
) -> dict:
    """
    Convert the state dict from a reference Mamba2 block to a Vollo Mamba2 block.
    """
    vollo_state_dict = {}

    # in_proj split: [z, x, B, C, dt]
    weight = ref_state_dict["in_proj.weight"]

    vollo_state_dict["proj_z.weight"] = weight[0:d_inner, :]
    vollo_state_dict["proj_x.weight"] = weight[d_inner : 2 * d_inner, :]
    vollo_state_dict["proj_B.weight"] = weight[2 * d_inner : 2 * d_inner + d_state, :]
    vollo_state_dict["proj_C.weight"] = weight[
        2 * d_inner + d_state : 2 * d_inner + 2 * d_state, :
    ]
    vollo_state_dict["proj_dt.weight"] = weight[2 * d_inner + 2 * d_state :, :]

    if "in_proj.bias" in ref_state_dict:
        bias = ref_state_dict["in_proj.bias"]
        vollo_state_dict["proj_z.bias"] = bias[0:d_inner]
        vollo_state_dict["proj_x.bias"] = bias[d_inner : 2 * d_inner]
        vollo_state_dict["proj_B.bias"] = bias[2 * d_inner : 2 * d_inner + d_state]
        vollo_state_dict["proj_C.bias"] = bias[
            2 * d_inner + d_state : 2 * d_inner + 2 * d_state
        ]
        vollo_state_dict["proj_dt.bias"] = bias[2 * d_inner + 2 * d_state :]

    # conv1d split
    conv_weight = ref_state_dict["conv1d.weight"]
    vollo_state_dict["conv_x.conv1d.conv.weight"] = conv_weight[0:d_inner, :, :]
    vollo_state_dict["conv_B.conv1d.conv.weight"] = conv_weight[
        d_inner : d_inner + d_state, :, :
    ]
    vollo_state_dict["conv_C.conv1d.conv.weight"] = conv_weight[
        d_inner + d_state :, :, :
    ]

    if "conv1d.bias" in ref_state_dict:
        conv_bias = ref_state_dict["conv1d.bias"]
        vollo_state_dict["conv_x.conv1d.conv.bias"] = conv_bias[0:d_inner]
        vollo_state_dict["conv_B.conv1d.conv.bias"] = conv_bias[
            d_inner : d_inner + d_state
        ]
        vollo_state_dict["conv_C.conv1d.conv.bias"] = conv_bias[d_inner + d_state :]

    # Simple mappings
    vollo_state_dict["dt_bias"] = ref_state_dict["dt_bias"]
    vollo_state_dict["A_log"] = ref_state_dict["A_log"]

    # D mapping: if D_has_hdim is False (default), D is [nheads]
    # Vollo D is always [d_inner]
    D = ref_state_dict["D"]
    if D.shape[0] == n_heads:
        # Repeat D for each head's dimension
        from einops import repeat

        vollo_state_dict["D"] = repeat(D, "h -> (h p)", p=d_inner // n_heads)
    else:
        vollo_state_dict["D"] = D

    vollo_state_dict["out_proj.weight"] = ref_state_dict["out_proj.weight"]
    if "out_proj.bias" in ref_state_dict:
        vollo_state_dict["out_proj.bias"] = ref_state_dict["out_proj.bias"]

    return vollo_state_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("d_model", [32, 64])
@pytest.mark.parametrize("d_state", [16, 32])
@pytest.mark.parametrize("d_conv", [2, 4])
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

    ref_model = (
        Mamba2Ref(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            bias=bias,
            conv_bias=conv_bias,
            rmsnorm=False,
        )
        .eval()
        .cuda()
    )

    vollo_model = (
        Mamba2Vollo(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=headdim,
            expand=expand,
            bias=bias,
            conv_bias=conv_bias,
            rmsnorm=False,
        )
        .eval()
        .cuda()
    )

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(
        ref_model.state_dict(), d_inner, d_state, n_heads
    )
    vollo_model.load_state_dict(vollo_state_dict, strict=True)

    T = 8
    batch_size = 1

    x = torch.randn(batch_size, T, d_model).cuda()

    # Reference output using step loop
    ref_outputs = []
    conv_state, ssm_state = ref_model.allocate_inference_cache(batch_size, T)
    conv_state = conv_state.cuda()
    ssm_state = ssm_state.cuda()

    with torch.no_grad():
        for i in range(T):
            step_out, conv_state, ssm_state = ref_model.step(
                x[:, i : i + 1, :], conv_state, ssm_state
            )
            ref_outputs.append(step_out)

        ref_out = torch.cat(ref_outputs, dim=1)

        # Vollo output
        vollo_out = vollo_model(x.squeeze(0))
        vollo_out = vollo_out.unsqueeze(0)

    # Check shapes
    assert ref_out.shape == vollo_out.shape

    # Check values
    np.testing.assert_allclose(
        ref_out.cpu().numpy(),
        vollo_out.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Mamba2 implementations output mismatch",
    )
