"""
The QEC LSTM decoder against a plain `torch.nn.LSTM` reference.

The zoo model splits the 2-layer `torch.nn.LSTM` into two 1-layer ones, so that
each can be placed on its own cores, and splits the head's concatenated Linear
in two; `convert_state_dict` maps trained weights across. What is checked here
is that this layout and its core placement compute the paper's network, both in
PyTorch and in the compiled Vollo program.
"""

import importlib

import numpy as np
import pytest
import torch
import vollo_torch
from beartype import beartype
from torch import nn

from vollo_model_zoo.vm import CONFIGS, _vollo_compile

qec_lstm = importlib.import_module("vollo_model_zoo.models.qec-lstm")


class Reference(nn.Module):
    """
    The decoder as written in the papers: a 2-layer `torch.nn.LSTM` over the
    round's defects, then a head on concat(relu(state), final_defects).
    """

    @beartype
    def __init__(self, num_defects: int, num_final: int, hidden_size: int):
        super().__init__()
        self.num_defects = num_defects
        self.lstm = nn.LSTM(num_defects, hidden_size, num_layers=2, batch_first=True)
        self.to_hidden = nn.Linear(hidden_size + num_final, hidden_size)
        self.logit = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Time, num_defects + num_final)

        Returns:
            Tensor of shape (Batch, Time, 1)
        """
        state, _ = self.lstm(x[..., : self.num_defects])
        joint = torch.cat([torch.relu(state), x[..., self.num_defects :]], dim=-1)
        return self.logit(torch.relu(self.to_hidden(joint)))


@beartype
def convert_state_dict(reference: Reference) -> dict[str, torch.Tensor]:
    """
    Convert the reference's state dict to the zoo model's: layer `l` of the
    2-layer LSTM becomes the 1-layer `lstm{l + 1}`, and the concatenated head
    Linear splits into one Linear per input.
    """
    ref = reference.state_dict()
    hidden_size = reference.lstm.hidden_size
    out = {}
    for layer in range(2):
        for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            out[f"lstm{layer + 1}.{name}_l0"] = ref[f"lstm.{name}_l{layer}"]
    out["state_to_hidden.weight"] = ref["to_hidden.weight"][:, :hidden_size]
    out["state_to_hidden.bias"] = ref["to_hidden.bias"]
    out["final_defects_to_hidden.weight"] = ref["to_hidden.weight"][:, hidden_size:]
    out["logit.weight"] = ref["logit.weight"]
    out["logit.bias"] = ref["logit.bias"]
    return out


# The core placements `_vm` picks: none, and the one for a small head
PLACEMENTS = {
    "compiler": {},
    "small_head": dict(lstm2_cores=[0, 1, 2], head_core=0),
}


@beartype
def build(
    distance: int, hidden_size: int, soft: bool, placement: str
) -> tuple[nn.Module, Reference]:
    torch.manual_seed(0)
    model = qec_lstm.QECLSTM(distance, hidden_size, soft=soft, **PLACEMENTS[placement])
    num_final = (distance**2 - 1) // 2
    reference = Reference(model.num_defects, num_final, hidden_size)
    # Non-trivial weights everywhere, biases included
    with torch.no_grad():
        for p in reference.parameters():
            p.uniform_(-0.3, 0.3)
    model.load_state_dict(convert_state_dict(reference), strict=True)
    return model.eval(), reference.eval()


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("distance,hidden_size", [(3, 16), (5, 32)])
@beartype
def test_matches_reference(distance: int, hidden_size: int, soft: bool, placement: str):
    model, reference = build(distance, hidden_size, soft, placement)
    x = torch.randn(2, 12, model.num_defects + (distance**2 - 1) // 2)

    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("config", ["V80", "IA-840f"])
@pytest.mark.parametrize("distance,hidden_size", [(3, 64), (5, 128)])
@beartype
def test_vm_placement_matches_reference(distance: int, hidden_size: int, config: str):
    """
    The compiled program, run one round per inference in the VM, against the
    reference, with everything left to the compiler and with each placement.
    """
    time = 16
    outputs = {}
    for placement in PLACEMENTS:
        model, reference = build(distance, hidden_size, False, placement)
        x = torch.randn(1, time, model.num_defects + (distance**2 - 1) // 2)
        program = _vollo_compile(
            model, x, time_axis=1, config=CONFIGS[config], allow_dynamic_weights=False
        )
        vm = program.to_vm(bit_accurate=False)
        outputs[placement] = vm.run_timesteps(x.numpy(), 1, 1)

        with torch.no_grad():
            expected = reference(x).numpy()
        # The VM without `bit_accurate` models the arithmetic in fp32
        np.testing.assert_allclose(outputs[placement], expected, atol=1e-5, rtol=1e-4)

    # Moving work between cores changes only how the VM's fp32 model of the
    # accumulators splits the sums
    for placement in list(PLACEMENTS)[1:]:
        np.testing.assert_allclose(
            outputs["compiler"], outputs[placement], atol=1e-6, rtol=1e-4
        )
