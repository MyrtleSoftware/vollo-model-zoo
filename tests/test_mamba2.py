import numpy as np
import pytest
import torch
from beartype import beartype
from fla_mamba2 import Mamba2 as FLAMamba2

from vollo_model_zoo.models.mamba2 import Mamba2 as VolloMamba2


@beartype
def convert_state_dict(
    fla_state_dict: dict, d_inner: int, d_state: int, n_heads: int
) -> dict:
    """
    Convert the state dict from an FLA Mamba2 block to a Vollo Mamba2 block.
    """
    vollo_state_dict = {}

    # in_proj split: [z, x, B, C, dt]
    # FLA's in_proj includes everything: 2 * d_inner + 2 * d_state + n_heads
    # Split as: [0, 0, d_inner, d_inner + 2 * d_state, n_heads] (where d_mlp is 0)
    # Actually, as deduced:
    # 0: d_inner (gate/z)
    # d_inner: 2 * d_inner (hidden/x)
    # 2 * d_inner: 2 * d_inner + d_state (B)
    # 2 * d_inner + d_state: 2 * d_inner + 2 * d_state (C)
    # 2 * d_inner + 2 * d_state : (dt)
    weight = fla_state_dict["in_proj.weight"]

    vollo_state_dict["proj_z.weight"] = weight[0:d_inner, :]
    vollo_state_dict["proj_x.weight"] = weight[d_inner : 2 * d_inner, :]
    vollo_state_dict["proj_B.weight"] = weight[2 * d_inner : 2 * d_inner + d_state, :]
    vollo_state_dict["proj_C.weight"] = weight[
        2 * d_inner + d_state : 2 * d_inner + 2 * d_state, :
    ]
    vollo_state_dict["proj_dt.weight"] = weight[2 * d_inner + 2 * d_state :, :]

    if "in_proj.bias" in fla_state_dict:
        bias = fla_state_dict["in_proj.bias"]
        vollo_state_dict["proj_z.bias"] = bias[0:d_inner]
        vollo_state_dict["proj_x.bias"] = bias[d_inner : 2 * d_inner]
        vollo_state_dict["proj_B.bias"] = bias[2 * d_inner : 2 * d_inner + d_state]
        vollo_state_dict["proj_C.bias"] = bias[
            2 * d_inner + d_state : 2 * d_inner + 2 * d_state
        ]
        vollo_state_dict["proj_dt.bias"] = bias[2 * d_inner + 2 * d_state :]

    # conv1d split
    # FLA conv1d is [conv_dim, 1, kernel_size]
    # conv_dim = d_inner + 2 * d_state
    conv_weight = fla_state_dict["conv1d.weight"]
    vollo_state_dict["conv_x.conv1d.conv.weight"] = conv_weight[0:d_inner, :, :]
    vollo_state_dict["conv_B.conv1d.conv.weight"] = conv_weight[
        d_inner : d_inner + d_state, :, :
    ]
    vollo_state_dict["conv_C.conv1d.conv.weight"] = conv_weight[
        d_inner + d_state :, :, :
    ]

    if "conv1d.bias" in fla_state_dict:
        conv_bias = fla_state_dict["conv1d.bias"]
        vollo_state_dict["conv_x.conv1d.conv.bias"] = conv_bias[0:d_inner]
        vollo_state_dict["conv_B.conv1d.conv.bias"] = conv_bias[
            d_inner : d_inner + d_state
        ]
        vollo_state_dict["conv_C.conv1d.conv.bias"] = conv_bias[d_inner + d_state :]

    # Simple mappings
    vollo_state_dict["dt_bias"] = fla_state_dict["dt_bias"]
    vollo_state_dict["A_log"] = fla_state_dict["A_log"]

    # D mapping: FLA D is [num_heads]
    # Vollo D is always [d_inner]
    D = fla_state_dict["D"]
    if D.shape[0] == n_heads:
        # Repeat D for each head's dimension
        from einops import repeat

        vollo_state_dict["D"] = repeat(D, "h -> (h p)", p=d_inner // n_heads)
    else:
        vollo_state_dict["D"] = D

    vollo_state_dict["out_proj.weight"] = fla_state_dict["out_proj.weight"]
    if "out_proj.bias" in fla_state_dict:
        vollo_state_dict["out_proj.bias"] = fla_state_dict["out_proj.bias"]

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
            rms_norm=False,  # We mock this if needed, but currently Vollo has no norm
            backend="triton",
        )
        .eval()
        .cuda()
    )

    # Mock the norm in FLA to match Vollo's "no norm" behavior (just gating)
    # Vollo does: y * act(z)
    # FLA norm(y, gate) usually does: (y / RMS(y)) * act(gate)
    # If we want to skip the norm, we just do y * act(gate)
    class MockRMSNormGated(torch.nn.Module):
        def __init__(self, act):
            super().__init__()
            self.act = act

        def forward(self, x, gate):
            return x * self.act(gate)

    fla_model.norm = MockRMSNormGated(fla_model.act).cuda()

    vollo_model = (
        VolloMamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=headdim,
            expand=expand,
            bias=bias,
            conv_bias=conv_bias,
            rmsnorm=False,
            activation="relu",
        )
        .eval()
        .cuda()
    )

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(
        fla_model.state_dict(), d_inner, d_state, n_heads
    )
    vollo_model.load_state_dict(vollo_state_dict, strict=True)

    T = 8
    batch_size = 1

    x = torch.randn(batch_size, T, d_model).cuda()

    with torch.no_grad():
        # FLA reference output
        # FLA expects [batch, seq_len, d_model]
        fla_out = fla_model(x)

        # Vollo output expects [seq_len, d_model]
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
