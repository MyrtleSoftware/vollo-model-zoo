"""
Stateless, two-entrypoint RNN-T model for the Vollo model zoo.

Unlike :mod:`rnnt`, this variant returns every recurrent state to the host.
The encoder/joint and prediction/joint entry points still share the final
joint network and are compiled into one Vollo program by :func:`main`.
"""

from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import Tensor, nn


class StatelessPredictionJoint(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        pred_n_hid: int,
        pred_rnn_layers: int,
        joint_n_hid: int,
        joint_network: nn.Module,
        fp8_weights: bool = False,
    ) -> None:
        """
        Prediction/joint entry point, with recurrent state held by the host.

        Every hidden and cell tensor is an explicit input and output, so the
        host owns all recurrent state between calls. That removes the
        compile-time limit on concurrent streams at the cost of extra I/O.

        The token embedding is a look-up, so the host performs it and passes the
        embedded vector in.

        Args:
               pred_n_hid:           Width of the prediction network's LSTM
               pred_rnn_layers:      Number of prediction LSTM layers
               joint_n_hid:          Width of the shared joint space
               joint_network:        Joint network, shared with the encoder so
                                     the compiler packs its weights once
               fp8_weights:          Store weight matrices as FP8 E4M3
        """
        super().__init__()

        self.fp8_weights = fp8_weights
        self.lstm = _ManualLSTMStack(pred_n_hid, pred_n_hid, pred_rnn_layers)
        self.joint_pred = nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        embed: Tensor,
        encoding: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run the prediction network and joint network for one token.

        Args:
            embed: ``[batch, pred_n_hid]`` host embedding look-up for the
                latest non-blank token.
            encoding: ``[batch, joint_n_hid]`` current encoder representation.
            h: ``[pred_rnn_layers, batch, pred_n_hid]`` hidden states.
            c: ``[pred_rnn_layers, batch, pred_n_hid]`` cell states.

        Returns:
            ``(prediction, logits, next_h, next_c)``. ``prediction`` is
            ``[batch, joint_n_hid]`` and ``logits`` is ``[batch, n_classes]``;
            the state tensors keep the shapes of ``h`` and ``c`` and must be
            supplied to the next prediction call.
        """
        with vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext():
            prediction, h, c = self.lstm(embed, h, c)
            prediction = self.joint_pred(prediction)
            logits = self.joint_network(encoding + prediction)

        return prediction, logits, h, c


class StatelessEncoderJoint(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        in_feats: int,
        enc_n_hid: int,
        enc_pre_rnn_layers: int,
        enc_post_rnn_layers: int,
        joint_n_hid: int,
        joint_network: nn.Module,
        fp8_weights: bool = False,
    ) -> None:
        """
        Encoder/joint entry point, with recurrent state held by the host.

        Args:
               in_feats:             Acoustic features per frame (two per call)
               enc_n_hid:            Width of the encoder's LSTMs
               enc_pre_rnn_layers:   Pre-RNN layers, advanced once per frame
               enc_post_rnn_layers:  Post-RNN layers, advanced once per pair
               joint_n_hid:          Width of the shared joint space
               joint_network:        Joint network, shared with the predictor so
                                     the compiler packs its weights once
               fp8_weights:          Store weight matrices as FP8 E4M3
        """
        super().__init__()

        self.fp8_weights = fp8_weights
        self.pre_rnn = _ManualLSTMStack(in_feats, enc_n_hid, enc_pre_rnn_layers)
        self.post_rnn = _ManualLSTMStack(2 * enc_n_hid, enc_n_hid, enc_post_rnn_layers)
        self.joint_enc = nn.Linear(enc_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        feats_0: Tensor,
        feats_1: Tensor,
        prediction: Tensor,
        pre_h: Tensor,
        pre_c: Tensor,
        post_h: Tensor,
        post_c: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Encode a pair of feature frames and compute joint logits.

        Taking two frames per call keeps the accelerator interface fixed-rate:
        the pre-RNN advances twice, then the post-RNN advances once per pair.

        Args:
            feats_0: ``[batch, in_feats]`` first frame of the pair.
            feats_1: ``[batch, in_feats]`` second frame of the pair.
            prediction: ``[batch, joint_n_hid]`` prediction representation.
            pre_h: ``[enc_pre_rnn_layers, batch, enc_n_hid]`` hidden states.
            pre_c: ``[enc_pre_rnn_layers, batch, enc_n_hid]`` cell states.
            post_h: ``[enc_post_rnn_layers, batch, enc_n_hid]`` hidden states.
            post_c: ``[enc_post_rnn_layers, batch, enc_n_hid]`` cell states.

        Returns:
            ``(encoding, logits, next_pre_h, next_pre_c, next_post_h,
            next_post_c)``. ``encoding`` is ``[batch, joint_n_hid]`` and
            ``logits`` is ``[batch, n_classes]``; the four state tensors keep
            their input shapes and must be supplied to the next encoder call.
        """
        with vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext():
            pre_0, pre_h, pre_c = self.pre_rnn(feats_0, pre_h, pre_c)
            pre_1, pre_h, pre_c = self.pre_rnn(feats_1, pre_h, pre_c)
            output, post_h, post_c = self.post_rnn(
                torch.cat((pre_0, pre_1), dim=-1),
                post_h,
                post_c,
            )
            encoding = self.joint_enc(output)
            logits = self.joint_network(encoding + prediction)

        return encoding, logits, pre_h, pre_c, post_h, post_c


class _ManualLSTMCell(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        gate_input_size = input_size + hidden_size
        self.input_gate = nn.Linear(gate_input_size, hidden_size)
        self.forget_gate = nn.Linear(gate_input_size, hidden_size)
        self.cell_gate = nn.Linear(gate_input_size, hidden_size)
        self.output_gate = nn.Linear(gate_input_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate_input = torch.cat((x, h), dim=-1)
        i = torch.sigmoid(self.input_gate(gate_input))
        f = torch.sigmoid(self.forget_gate(gate_input))
        g = torch.tanh(self.cell_gate(gate_input))
        o = torch.sigmoid(self.output_gate(gate_input))

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class _ManualLSTMStack(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.cells = nn.ModuleList(
            [
                _ManualLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_h = []
        next_c = []
        for layer, cell in enumerate(self.cells):
            h_layer, c_layer = cell(x, h[layer], c[layer])
            next_h.append(h_layer)
            next_c.append(c_layer)
            x = h_layer
        return x, torch.stack(next_h), torch.stack(next_c)


@beartype
def multi_model_entries(
    *,
    n_classes: int,
    pred_n_hid: int,
    pred_rnn_layers: int,
    in_feats: int,
    enc_n_hid: int,
    enc_pre_rnn_layers: int,
    enc_post_rnn_layers: int,
    joint_n_hid: int,
    fp8_weights: bool,
) -> list:
    """
    Construct the two entry points with one shared joint network.
    """
    from vollo_model_zoo.vm import MultiModelEntry

    joint_network = nn.Sequential(
        nn.ReLU(),
        nn.Linear(joint_n_hid, n_classes),
    )
    prediction = StatelessPredictionJoint(
        pred_n_hid=pred_n_hid,
        pred_rnn_layers=pred_rnn_layers,
        joint_n_hid=joint_n_hid,
        fp8_weights=fp8_weights,
        joint_network=joint_network,
    )
    encoder = StatelessEncoderJoint(
        in_feats=in_feats,
        enc_n_hid=enc_n_hid,
        enc_pre_rnn_layers=enc_pre_rnn_layers,
        enc_post_rnn_layers=enc_post_rnn_layers,
        joint_n_hid=joint_n_hid,
        fp8_weights=fp8_weights,
        joint_network=joint_network,
    )

    # There is no Scan and no streaming axis: one call is one time step, and the
    # state tensors below describe shapes only.
    batch = 1
    pred_state = torch.zeros(pred_rnn_layers, batch, pred_n_hid)
    pre_state = torch.zeros(enc_pre_rnn_layers, batch, enc_n_hid)
    post_state = torch.zeros(enc_post_rnn_layers, batch, enc_n_hid)

    return [
        MultiModelEntry(
            name="predictor",
            model=prediction,
            inputs=(
                torch.randn(batch, pred_n_hid),  # embed
                torch.randn(batch, joint_n_hid),  # output from encoder/joint
                pred_state,
                pred_state.clone(),
            ),
            meta={
                "hidden": pred_n_hid,
                "joint_hidden": joint_n_hid,
            },
        ),
        MultiModelEntry(
            name="encoder",
            model=encoder,
            inputs=(
                torch.randn(batch, in_feats),  # feats_0
                torch.randn(batch, in_feats),  # feats_1
                torch.randn(batch, joint_n_hid),  # output from prediction/joint
                pre_state,
                pre_state.clone(),
                post_state,
                post_state.clone(),
            ),
            meta={
                "hidden": enc_n_hid,
                "joint_hidden": joint_n_hid,
            },
        ),
    ]


@beartype
def _vm(
    *,
    n_classes: int,
    pred_n_hid: int,
    pred_rnn_layers: int,
    in_feats: int,
    enc_n_hid: int,
    enc_pre_rnn_layers: int,
    enc_post_rnn_layers: int,
    joint_n_hid: int,
    fp8_weights: bool,
    config: str,
) -> list:
    from vollo_model_zoo.vm import vollo_multi_model_info

    return vollo_multi_model_info(
        multi_model_entries(
            n_classes=n_classes,
            pred_n_hid=pred_n_hid,
            pred_rnn_layers=pred_rnn_layers,
            in_feats=in_feats,
            enc_n_hid=enc_n_hid,
            enc_pre_rnn_layers=enc_pre_rnn_layers,
            enc_post_rnn_layers=enc_post_rnn_layers,
            joint_n_hid=joint_n_hid,
            fp8_weights=fp8_weights,
        ),
        config=config,
    )


@beartype
def main(config: str = "V80") -> Generator:
    models = [
        # 1M baseline used by the model-zoo test suite.
        dict(
            n_classes=256,
            pred_n_hid=192,
            pred_rnn_layers=1,
            in_feats=40,
            enc_n_hid=176,
            enc_pre_rnn_layers=1,
            enc_post_rnn_layers=1,
            joint_n_hid=288,
            fp8_weights=False,
        ),
        # 23M model for BF16/FP32
        dict(
            n_classes=1024,
            pred_n_hid=424,
            pred_rnn_layers=2,
            in_feats=240,
            enc_n_hid=680,
            enc_pre_rnn_layers=2,
            enc_post_rnn_layers=3,
            joint_n_hid=256,
            fp8_weights=False,
        ),
        # 49M model via fp8 weights
        dict(
            n_classes=1024,
            pred_n_hid=512,
            pred_rnn_layers=2,
            in_feats=240,
            enc_n_hid=1024,
            enc_pre_rnn_layers=2,
            enc_post_rnn_layers=3,
            joint_n_hid=512,
            fp8_weights=True,
        ),
    ]

    for x in models:
        yield from _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
