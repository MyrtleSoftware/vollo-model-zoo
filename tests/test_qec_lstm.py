"""
The QEC LSTM decoder against a plain `torch.nn.LSTM` reference.

The zoo model splits the 2-layer `torch.nn.LSTM` into two 1-layer ones, so that
each can be placed on its own cores, but keeps the 2-layer state-dict layout;
and it splits the head's concatenated Linear in two, which `convert_state_dict`
maps across. What is checked here is that this layout and its core placement
compute the paper's network, both in PyTorch and in the compiled Vollo program,
and that checkpoints of the 2-layer layout load.
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
    Convert the reference's state dict to the zoo model's: the 2-layer LSTM's
    keys are kept (the model loads them), and the concatenated head Linear
    splits into one Linear per input.
    """
    ref = reference.state_dict()
    hidden_size = reference.lstm.hidden_size
    out = {k: v for k, v in ref.items() if k.startswith("lstm.")}
    out["state_to_hidden.weight"] = ref["to_hidden.weight"][:, :hidden_size]
    out["state_to_hidden.bias"] = ref["to_hidden.bias"]
    out["final_defects_to_hidden.weight"] = ref["to_hidden.weight"][:, hidden_size:]
    out["logit.weight"] = ref["logit.weight"]
    out["logit.bias"] = ref["logit.bias"]
    return out


# The core placements `_vm` picks: none, the one for a small head, and the one
# for a big head (on 6 cores, with 8 LSTM hidden blocks)
PLACEMENTS = {
    "compiler": {},
    "small_head": dict(lstm2_cores=[0, 1, 2], head_cores=[0]),
    "big_head": dict(lstm1_cores=[0, 1, 3, 4, 5], head_cores=[2, 3, 4, 5]),
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
@pytest.mark.parametrize("distance,hidden_size", [(3, 64), (5, 128), (11, 256)])
@beartype
def test_vm_placement_matches_reference(distance: int, hidden_size: int, config: str):
    """
    The compiled program, run one round per inference in the VM, against the
    reference, with everything left to the compiler and with each placement
    that fits the config's cores.
    """
    num_cores = CONFIGS[config].num_cores
    placements = [
        name
        for name, cores in PLACEMENTS.items()
        if all(c < num_cores for v in cores.values() for c in v)
    ]
    time = 16
    outputs = {}
    for placement in placements:
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
    for placement in placements[1:]:
        np.testing.assert_allclose(
            outputs["compiler"], outputs[placement], atol=1e-6, rtol=1e-4
        )


class TwoLayerDecoder(nn.Module):
    """
    The decoder before its LSTM layers were split for placement: the same
    network on one 2-layer `nn.LSTM`, with the zoo model's state-dict layout.
    """

    @beartype
    def __init__(self, distance: int, hidden_size: int, soft: bool):
        super().__init__()
        stabilisers = distance**2 - 1
        self.num_defects = stabilisers * (2 if soft else 1)
        self.lstm = nn.LSTM(
            self.num_defects, hidden_size, num_layers=2, batch_first=True
        )
        self.state_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.final_defects_to_hidden = nn.Linear(
            stabilisers // 2, hidden_size, bias=False
        )
        self.logit = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, num_defects + (d^2 - 1) / 2] -> [B, T, 1]"""
        state, _ = self.lstm(x[..., : self.num_defects])
        hidden = self.state_to_hidden(torch.relu(state))
        hidden = hidden + self.final_defects_to_hidden(x[..., self.num_defects :])
        return self.logit(torch.relu(hidden))


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
@beartype
def test_two_layer_checkpoint_round_trips(soft: bool, placement: str):
    """
    A checkpoint of the 2-layer decoder loads into the zoo model (strictly) and
    gives the same outputs; the zoo model saves the same keys, in the same
    order, and its checkpoint loads back into the 2-layer decoder.
    """
    torch.manual_seed(0)
    two_layer = TwoLayerDecoder(5, 32, soft)
    with torch.no_grad():
        for p in two_layer.parameters():
            p.uniform_(-0.3, 0.3)
    model = qec_lstm.QECLSTM(5, 32, soft=soft, **PLACEMENTS[placement])
    model.load_state_dict(two_layer.state_dict())
    x = torch.randn(2, 12, model.num_defects + 12)
    with torch.no_grad():
        torch.testing.assert_close(model(x), two_layer(x), rtol=1e-6, atol=1e-6)

    saved = model.state_dict()
    assert list(saved) == list(two_layer.state_dict())
    reloaded = TwoLayerDecoder(5, 32, soft)
    reloaded.load_state_dict(saved)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), two_layer(x), rtol=0, atol=0)


@pytest.mark.parametrize("placement", list(PLACEMENTS))
@beartype
def test_wrong_size_tensor_is_reported_by_its_checkpoint_name(placement: str):
    state = TwoLayerDecoder(5, 32, False).state_dict()
    state["lstm.weight_hh_l1"] = state["lstm.weight_hh_l1"][:-1]
    model = qec_lstm.QECLSTM(5, 32, **PLACEMENTS[placement])
    before = model.lstm2.weight_hh_l0.clone()
    with pytest.raises(
        RuntimeError, match=r"size mismatch for lstm\.weight_hh_l1"
    ) as e:
        model.load_state_dict(state)
    assert "lstm2" not in str(e.value)
    torch.testing.assert_close(model.lstm2.weight_hh_l0, before, rtol=0, atol=0)


@pytest.mark.parametrize("placement", list(PLACEMENTS))
@beartype
def test_missing_tensor_is_reported_by_its_checkpoint_name(placement: str):
    state = TwoLayerDecoder(5, 32, False).state_dict()
    del state["lstm.weight_ih_l1"]
    model = qec_lstm.QECLSTM(5, 32, **PLACEMENTS[placement])
    with pytest.raises(RuntimeError, match=r"lstm\.weight_ih_l1") as e:
        model.load_state_dict(state)
    assert "lstm2" not in str(e.value)
    result = model.load_state_dict(state, strict=False)
    assert result.missing_keys == ["lstm.weight_ih_l1"]
    assert result.unexpected_keys == []


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("distance,hidden_size", [(3, 16), (5, 32)])
@beartype
def test_parts_compose_to_decoder(distance: int, hidden_size: int, soft: bool):
    """
    The head-once deviation: `QECLSTMHead` on each round's `QECLSTMRounds`
    state and final defects is the decoder's logit for that round.
    """
    model, reference = build(distance, hidden_size, soft, "compiler")
    rounds, head = qec_lstm.QECLSTMRounds(model), qec_lstm.QECLSTMHead(model)
    x = torch.randn(2, 12, model.num_defects + (distance**2 - 1) // 2)

    with torch.no_grad():
        state = rounds(x[..., : model.num_defects])
        logits = head(torch.cat([state, x[..., model.num_defects :]], dim=-1))
        torch.testing.assert_close(logits, reference(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("config", ["V80", "IA-840f"])
@pytest.mark.parametrize("distance,hidden_size", [(3, 64), (5, 128)])
@beartype
def test_vm_parts_match_reference(distance: int, hidden_size: int, config: str):
    """
    The two programs in the VM: stream the rounds program over every round,
    then run the head program once on the last round's state and the final
    defects, against the reference's logit after the last round.
    """
    model, reference = build(distance, hidden_size, False, "compiler")
    x = torch.randn(1, 16, model.num_defects + (distance**2 - 1) // 2)
    defects = x[..., : model.num_defects]

    rounds = _vollo_compile(
        qec_lstm.QECLSTMRounds(model), defects, time_axis=1, config=CONFIGS[config]
    )
    states = rounds.to_vm(bit_accurate=False).run_timesteps(defects.numpy(), 1, 1)

    head_input = np.concatenate([states[:, -1], x[:, -1, model.num_defects :]], -1)
    head_input = head_input.astype(np.float32)
    head = _vollo_compile(
        qec_lstm.QECLSTMHead(model),
        torch.from_numpy(head_input),
        time_axis=None,
        config=CONFIGS[config],
    )
    logit = head.to_vm(bit_accurate=False).run(head_input)

    with torch.no_grad():
        expected = reference(x)[:, -1].numpy()
    np.testing.assert_allclose(logit, expected, atol=1e-5, rtol=1e-4)
