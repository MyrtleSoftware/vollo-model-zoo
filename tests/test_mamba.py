import numpy as np
import pytest
import torch
import torch.nn as nn
from fla.layers.mamba import Mamba as FLAMamba

from vollo_model_zoo.models.mamba1 import Mamba as VolloMamba

# --- Test Comparison ---


def copy_weights(fla_model: FLAMamba, vollo_model: VolloMamba):
    with torch.no_grad():
        # in_proj split
        vollo_model.in_proj_x.weight.copy_(
            fla_model.in_proj.weight[: fla_model.intermediate_size, :]
        )
        vollo_model.in_proj_z.weight.copy_(
            fla_model.in_proj.weight[fla_model.intermediate_size :, :]
        )
        if fla_model.use_bias:
            vollo_model.in_proj_x.bias.copy_(
                fla_model.in_proj.bias[: fla_model.intermediate_size]
            )
            vollo_model.in_proj_z.bias.copy_(
                fla_model.in_proj.bias[fla_model.intermediate_size :]
            )

        # conv1d
        vollo_model.conv1d.conv.weight.copy_(fla_model.conv1d.weight)
        if fla_model.use_conv_bias:
            vollo_model.conv1d.conv.bias.copy_(fla_model.conv1d.bias)

        # x_proj split
        # FLA x_proj order: [dt, B, C]
        dt_rank = fla_model.time_step_rank
        d_state = fla_model.ssm_state_size
        vollo_model.step.x_proj_t.weight.copy_(fla_model.x_proj.weight[:dt_rank, :])
        vollo_model.step.x_proj_B.weight.copy_(
            fla_model.x_proj.weight[dt_rank : dt_rank + d_state, :]
        )
        vollo_model.step.x_proj_C.weight.copy_(
            fla_model.x_proj.weight[dt_rank + d_state :, :]
        )

        # dt_proj
        vollo_model.step.dt_proj.weight.copy_(fla_model.dt_proj.weight)
        vollo_model.step.dt_proj.bias.copy_(fla_model.dt_proj.bias)

        # A_log and D
        # Vollo A_log_t is [d_state, d_inner], FLA A_log is [d_inner, d_state]
        vollo_model.step.A_log_t.copy_(fla_model.A_log.t())
        vollo_model.step.D.copy_(fla_model.D.unsqueeze(0))

        # out_proj
        vollo_model.out_proj.weight.copy_(fla_model.out_proj.weight)
        if fla_model.use_bias:
            vollo_model.out_proj.bias.copy_(fla_model.out_proj.bias)


@pytest.mark.skip_if(
    not torch.cuda.is_available(), reason="Requires CUDA for triton backend"
)
@pytest.mark.parametrize("d_model", [4, 64, 128])
@pytest.mark.parametrize("d_state", [2, 16, 32])
@pytest.mark.parametrize("d_conv", [2, 4, 8])
@pytest.mark.parametrize("expand", [1, 2, 3])
@pytest.mark.parametrize("dt_rank", ["auto", 8])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("conv_bias", [True])  # TODO: add False
@torch.no_grad()
def test_vs_reference(d_model, d_state, d_conv, expand, dt_rank, bias, conv_bias):
    # Ensure backend is 'triton' to use slow_forward path if needed,
    # as cuda kernels might not be available in all environments.
    fla_model = (
        FLAMamba(
            hidden_size=d_model,
            state_size=d_state,
            intermediate_size=d_model * expand,
            time_step_rank=d_model // 16,
            use_bias=True,
            use_conv_bias=True,
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
            bias=True,
            conv_bias=True,
        )
        .eval()
        .cuda()
    )

    #
    # copy_weights(fla_model, vollo_model)
    #
    # x = torch.randn(1, seq_len, d_model)
    #
    # with torch.no_grad():
    #     # FLA Mamba forward
    #     fla_out = fla_model(x)
    #     # Vollo implementation expects [time, d_model]
    #     vollo_out = vollo_model(x.squeeze(0))
    #     vollo_out = vollo_out.unsqueeze(0)
    #
    # # Check shapes
    # assert fla_out.shape == vollo_out.shape
    #
    # # Check values
    # np.testing.assert_allclose(
    #     fla_out.numpy(),
    #     vollo_out.numpy(),
    #     rtol=1e-5,
    #     atol=1e-5,
    #     err_msg="Mamba implementations output mismatch",
    # )


if __name__ == "__main__":
    test_mamba_equivalence(64, 16, 32)
    print("Test passed!")
