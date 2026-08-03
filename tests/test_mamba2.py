import numpy as np
import pytest
import torch
from beartype import beartype
from fla.layers.mamba2 import Mamba2 as FLAMamba2

from vollo_model_zoo.models.mamba2 import Mamba2 as VolloMamba2


@beartype
def convert_state_dict(fla_state_dict: dict) -> dict:
    """
    Convert the state dict from an FLA Mamba2 block to a Vollo Mamba2 block.
    """
    vollo_state_dict = {}

    # in_proj split: [z, x, B, C, dt]
    weight = fla_state_dict["in_proj.weight"]

    # We need to find d_inner and d_state.
    # A_log has shape [n_heads]
    n_heads = fla_state_dict["A_log"].shape[0]

    # out_proj has shape [hidden_size, d_inner]
    d_inner = fla_state_dict["out_proj.weight"].shape[1]

    # projection_size = 2 * d_inner + 2 * d_state + n_heads
    projection_size = weight.shape[0]
    d_state = (projection_size - 2 * d_inner - n_heads) // 2

    n = d_inner

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
        # Vollo D is [d_inner]
        from einops import repeat

        vollo_state_dict["D"] = repeat(D, "h -> (h p)", p=n // n_heads)
    else:
        vollo_state_dict["D"] = D

    # Norm
    if "norm.weight" in fla_state_dict:
        vollo_state_dict["norm.weight"] = fla_state_dict["norm.weight"]

    vollo_state_dict["out_proj.weight"] = fla_state_dict["out_proj.weight"]
    if "out_proj.bias" in fla_state_dict:
        vollo_state_dict["out_proj.bias"] = fla_state_dict["out_proj.bias"]

    return vollo_state_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for triton")
@pytest.mark.parametrize("d_model", [32, 64])
@pytest.mark.parametrize("d_state", [16, 32])
@pytest.mark.parametrize("d_conv", [2, 4])
@pytest.mark.parametrize("expand", [1, 2])
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
            # These sizes have as few as 2 heads, too few to partition; the
            # partitioned path is checked against this one in the test below.
            head_partitions=None,
        )
        .eval()
        .cuda()
    )

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(fla_model.state_dict())

    vollo_model.load_state_dict(vollo_state_dict, strict=True)

    T = 32

    x = torch.randn(1, T, d_model).cuda()

    with torch.no_grad():
        # FLA Mamba2 forward
        fla_out = fla_model(x)
        # Vollo implementation expects [time, d_model]
        vollo_out = vollo_model(x.squeeze(0))
        vollo_out = vollo_out.unsqueeze(0)

    if isinstance(fla_out, tuple):
        # Some FLA versions return a tuple that needs destructuring
        fla_out, _, _ = fla_out

    # Check shapes
    assert fla_out.shape == vollo_out.shape

    for i in range(T):
        np.testing.assert_allclose(
            fla_out[0, i].cpu().numpy(),
            vollo_out[0, i].cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Mamba2 implementations output mismatch at step={i}",
        )


@pytest.mark.filterwarnings("ignore:n_heads")  # 5 splits 12 heads unevenly, on purpose
@pytest.mark.parametrize("head_partitions", [1, 2, 5, 6])
def test_mamba2_partitioned_matches_unpartitioned(head_partitions: int):
    torch.manual_seed(42)

    d_model = 96

    def build(partitions):
        return VolloMamba2(
            d_model=d_model, expand=2, d_head=16, head_partitions=partitions
        )

    # Partitioned -> wide keys, then wide -> partitioned keys again.
    partitioned = build(head_partitions)
    unpartitioned = build(None)
    unpartitioned.load_state_dict(partitioned.state_dict(), strict=True)
    reloaded = build(head_partitions)
    reloaded.load_state_dict(unpartitioned.state_dict(), strict=True)

    x = torch.randn(16, d_model)

    with torch.no_grad():
        expected = partitioned(x).numpy()
        for name, model in [("unpartitioned", unpartitioned), ("reloaded", reloaded)]:
            np.testing.assert_allclose(
                expected,
                model(x).numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"{name} does not match the partitioned model",
            )


def test_mamba2_head_partitions_validation():
    # 96 * 2 / 16 = 12 heads
    kwargs = dict(d_model=96, expand=2, d_head=16)

    with pytest.raises(ValueError, match="must not exceed the number of heads"):
        VolloMamba2(**kwargs, head_partitions=13)

    with pytest.raises(ValueError, match="must be >= 1"):
        VolloMamba2(**kwargs, head_partitions=0)

    with pytest.warns(UserWarning, match="not a multiple of head_partitions"):
        VolloMamba2(**kwargs, head_partitions=5)
