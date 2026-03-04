import numpy as np
import pytest
import torch
from fla.layers.mamba import Mamba as FLAMamba

from vollo_model_zoo.models.mamba1 import Mamba as VolloMamba

# --- Weight Conversion ---


def convert_state_dict(
    fla_state_dict,
    intermediate_size,
    ssm_state_size,
    time_step_rank,
    use_bias,
    use_conv_bias,
):
    vollo_state_dict = {}

    # in_proj split: FLA has one Linear, Vollo has two (in_proj_x and in_proj_z)
    weight = fla_state_dict["in_proj.weight"]
    vollo_state_dict["in_proj_x.weight"] = weight[:intermediate_size, :]
    vollo_state_dict["in_proj_z.weight"] = weight[intermediate_size:, :]

    if use_bias:
        bias = fla_state_dict["in_proj.bias"]
        vollo_state_dict["in_proj_x.bias"] = bias[:intermediate_size]
        vollo_state_dict["in_proj_z.bias"] = bias[intermediate_size:]

    # conv1d: Vollo's PaddedConv1d wraps an nn.Conv1d in self.conv
    vollo_state_dict["conv1d.conv.weight"] = fla_state_dict["conv1d.weight"]
    if use_conv_bias:
        vollo_state_dict["conv1d.conv.bias"] = fla_state_dict["conv1d.bias"]

    # x_proj split: FLA is [dt_rank + 2 * ssm_state_size, intermediate_size]
    # Vollo has three separate linear layers
    x_proj_weight = fla_state_dict["x_proj.weight"]
    vollo_state_dict["step.x_proj_t.weight"] = x_proj_weight[:time_step_rank, :]
    vollo_state_dict["step.x_proj_B.weight"] = x_proj_weight[
        time_step_rank : time_step_rank + ssm_state_size, :
    ]
    vollo_state_dict["step.x_proj_C.weight"] = x_proj_weight[
        time_step_rank + ssm_state_size :, :
    ]

    # dt_proj
    vollo_state_dict["step.dt_proj.weight"] = fla_state_dict["dt_proj.weight"]
    vollo_state_dict["step.dt_proj.bias"] = fla_state_dict["dt_proj.bias"]

    # A_log and D
    # Vollo A_log_t is [ssm_state_size, intermediate_size], FLA A_log is [intermediate_size, ssm_state_size]
    vollo_state_dict["step.A_log_t"] = fla_state_dict["A_log"].t()
    vollo_state_dict["step.D"] = fla_state_dict["D"].unsqueeze(0)

    # out_proj
    vollo_state_dict["out_proj.weight"] = fla_state_dict["out_proj.weight"]
    if use_bias:
        vollo_state_dict["out_proj.bias"] = fla_state_dict["out_proj.bias"]

    return vollo_state_dict


# --- Test Comparison ---


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA for triton backend"
)
@pytest.mark.parametrize("d_model", [64])
@pytest.mark.parametrize("d_state", [16])
@pytest.mark.parametrize("d_conv", [4])
@pytest.mark.parametrize("expand", [2])
@pytest.mark.parametrize("dt_rank", [16])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("conv_bias", [True])
def test_mamba_equivalence(
    d_model,
    d_state,
    d_conv,
    expand,
    dt_rank,
    bias,
    conv_bias,
):
    fla_model = (
        FLAMamba(
            hidden_size=d_model,
            state_size=d_state,
            conv_kernel=d_conv,
            intermediate_size=d_model * expand,
            time_step_rank=dt_rank,
            use_bias=bias,
            use_conv_bias=conv_bias,
            backend="triton",
        )
        .eval()
        .cuda()
    )

    vollo_model = (
        VolloMamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            conv_bias=conv_bias,
        )
        .eval()
        .cuda()
    )

    # Convert and load state dict
    vollo_state_dict = convert_state_dict(
        fla_model.state_dict(),
        intermediate_size=d_model * expand,
        ssm_state_size=d_state,
        time_step_rank=dt_rank,
        use_bias=bias,
        use_conv_bias=conv_bias,
    )

    vollo_model.load_state_dict(vollo_state_dict, strict=False)

    x = torch.randn(1, 32, d_model).cuda()

    with torch.no_grad():
        # FLA Mamba forward
        fla_out = fla_model(x)
        # Vollo implementation expects [time, d_model]
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
        err_msg="Mamba implementations output mismatch",
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_mamba_equivalence(64, 2, 16, 32)
        print("Test passed!")
    else:
        print("CUDA not available, skipping test.")
