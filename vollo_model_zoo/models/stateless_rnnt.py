"""Stateless, two-entrypoint RNN-T model for the Vollo model zoo.

Unlike :mod:`rnnt`, this variant returns every recurrent state to the host.
The encoder/joint and prediction/joint entry points still share the final
joint network and are compiled into one Vollo program by :func:`main`.
"""

from collections.abc import Generator
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from packaging.version import Version
from torch import Tensor, nn


class _ManualLSTMCell(nn.Module):
    """One explicit-state LSTM cell expressed with Vollo-supported operators."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        gate_input_size = input_size + hidden_size
        self.input_gate = nn.Linear(gate_input_size, hidden_size)
        self.forget_gate = nn.Linear(gate_input_size, hidden_size)
        self.cell_gate = nn.Linear(gate_input_size, hidden_size)
        self.output_gate = nn.Linear(gate_input_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Run one explicit-state LSTM cell.

        Args:
            x: ``[batch, input_size]`` input values.
            h: ``[batch, hidden_size]`` hidden state.
            c: ``[batch, hidden_size]`` cell state.

        Returns:
            A pair ``(next_h, next_c)``; both tensors have shape
            ``[batch, hidden_size]``.
        """
        gate_input = torch.cat((x, h), dim=-1)
        i = torch.sigmoid(self.input_gate(gate_input))
        f = torch.sigmoid(self.forget_gate(gate_input))
        g = torch.tanh(self.cell_gate(gate_input))
        o = torch.sigmoid(self.output_gate(gate_input))

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class _ManualLSTMStack(nn.Module):
    """Apply a stack of explicit-state LSTM cells to one time step."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [
                _ManualLSTMCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one time step through every explicit-state LSTM layer.

        Args:
            x: ``[batch, input_size]`` input to the first layer.
            h: ``[num_layers, batch, hidden_size]`` hidden states.
            c: ``[num_layers, batch, hidden_size]`` cell states.

        Returns:
            A tuple ``(output, next_h, next_c)`` where ``output`` is
            ``[batch, hidden_size]`` and both state tensors are
            ``[num_layers, batch, hidden_size]``.
        """
        next_h = []
        next_c = []
        for layer, cell in enumerate(self.cells):
            h_layer, c_layer = cell(x, h[layer], c[layer])
            next_h.append(h_layer)
            next_c.append(c_layer)
            x = h_layer
        return x, torch.stack(next_h), torch.stack(next_c)


class StatelessEncoderJoint(nn.Module):
    """Run one encoder step and return all recurrent state to the host."""

    def __init__(
        self,
        *,
        in_feats: int,
        enc_n_hid: int,
        enc_pre_rnn_layers: int,
        enc_post_rnn_layers: int,
        joint_n_hid: int,
        fp8_weights: bool = False,
        joint_network: nn.Module,
    ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.fp8_weights = fp8_weights
        self.pre_rnn = _ManualLSTMStack(
            in_feats,
            enc_n_hid,
            enc_pre_rnn_layers,
        )
        self.post_rnn = _ManualLSTMStack(
            2 * enc_n_hid,
            enc_n_hid,
            enc_post_rnn_layers,
        )
        self.joint_enc = nn.Linear(enc_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        x: Tensor,
        pre_h: Tensor,
        pre_c: Tensor,
        post_h: Tensor,
        post_c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode one stacked pair of feature frames and compute joint logits.

        Args:
            x: ``[batch, 2 * in_feats + joint_n_hid]``. The first two slices
                are consecutive acoustic feature frames; the final slice is
                the current prediction representation.
            pre_h: ``[enc_pre_rnn_layers, batch, enc_n_hid]`` hidden states.
            pre_c: ``[enc_pre_rnn_layers, batch, enc_n_hid]`` cell states.
            post_h: ``[enc_post_rnn_layers, batch, enc_n_hid]`` hidden states.
            post_c: ``[enc_post_rnn_layers, batch, enc_n_hid]`` cell states.

        Returns:
            ``(output, next_pre_h, next_pre_c, next_post_h, next_post_c)``.
            ``output`` is ``[batch, joint_n_hid + n_classes]`` and
            concatenates the encoder representation with the joint logits.
            The four state tensors retain their corresponding input shapes
            and must be supplied to the next encoder call.
        """
        features = x[..., : 2 * self.in_feats]
        prediction = x[..., 2 * self.in_feats :]
        feature_0 = features[:, : self.in_feats]
        feature_1 = features[:, self.in_feats :]

        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            pre_0, pre_h, pre_c = self.pre_rnn(feature_0, pre_h, pre_c)
            pre_1, pre_h, pre_c = self.pre_rnn(feature_1, pre_h, pre_c)
            output, post_h, post_c = self.post_rnn(
                torch.cat((pre_0, pre_1), dim=-1),
                post_h,
                post_c,
            )
            encoding = self.joint_enc(output)
            logits = self.joint_network(encoding + prediction)

        return (
            torch.cat((encoding, logits), dim=-1),
            pre_h,
            pre_c,
            post_h,
            post_c,
        )


class StatelessPredictionJoint(nn.Module):
    """Run one prediction step and return recurrent state to the host."""

    def __init__(
        self,
        *,
        n_classes: int,
        pred_n_hid: int,
        pred_rnn_layers: int,
        joint_n_hid: int,
        fp8_weights: bool = False,
        joint_network: nn.Module,
    ) -> None:
        super().__init__()
        self.vocab_without_blank = n_classes - 1
        self.fp8_weights = fp8_weights
        self.embedding = nn.Linear(
            self.vocab_without_blank,
            pred_n_hid,
            bias=False,
        )
        self.lstm = _ManualLSTMStack(
            pred_n_hid,
            pred_n_hid,
            pred_rnn_layers,
        )
        self.joint_pred = nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the prediction network and joint network for one token.

        Args:
            x: ``[batch, (n_classes - 1) + joint_n_hid]``. The first slice is
                a one-hot non-blank token; the second is the current encoder
                representation.
            h: ``[pred_rnn_layers, batch, pred_n_hid]`` hidden states.
            c: ``[pred_rnn_layers, batch, pred_n_hid]`` cell states.

        Returns:
            A tuple ``(output, next_h, next_c)``. ``output`` is
            ``[batch, joint_n_hid + n_classes]`` and concatenates the new
            prediction representation with the joint logits. The state
            tensors retain the shapes of ``h`` and ``c``.
        """
        token = x[..., : self.vocab_without_blank]
        encoding = x[..., self.vocab_without_blank :]
        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            embedded = self.embedding(token)
            prediction, next_h, next_c = self.lstm(embedded, h, c)
            prediction = self.joint_pred(prediction)
            logits = self.joint_network(encoding + prediction)

        return torch.cat((prediction, logits), dim=-1), next_h, next_c


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
    """Compile the explicit-state prediction and encoder entry points."""
    from vollo_model_zoo.vm import MultiModelEntry, vollo_multi_model_info

    joint_network = nn.Sequential(
        nn.ReLU(),
        nn.Linear(joint_n_hid, n_classes),
    )
    prediction = StatelessPredictionJoint(
        n_classes=n_classes,
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

    prediction_input = torch.randn(1, n_classes - 1 + joint_n_hid)
    prediction_h = torch.zeros(pred_rnn_layers, 1, pred_n_hid)
    prediction_c = torch.zeros_like(prediction_h)
    encoder_input = torch.randn(1, 2 * in_feats + joint_n_hid)
    encoder_pre_h = torch.zeros(enc_pre_rnn_layers, 1, enc_n_hid)
    encoder_pre_c = torch.zeros_like(encoder_pre_h)
    encoder_post_h = torch.zeros(enc_post_rnn_layers, 1, enc_n_hid)
    encoder_post_c = torch.zeros_like(encoder_post_h)

    return vollo_multi_model_info(
        [
            MultiModelEntry(
                name="prediction_joint",
                model=prediction,
                inputs=(prediction_input, prediction_h, prediction_c),
            ),
            MultiModelEntry(
                name="encoder_joint",
                model=encoder,
                inputs=(
                    encoder_input,
                    encoder_pre_h,
                    encoder_pre_c,
                    encoder_post_h,
                    encoder_post_c,
                ),
            ),
        ],
        config=config,
        meta={
            "enc_n_hid": enc_n_hid,
            "joint_n_hid": joint_n_hid,
            "pred_n_hid": pred_n_hid,
        },
    )


@beartype
def main(config: str = "V80") -> Generator:
    models = [
        # Small baseline used by the model-zoo test suite.
        dict(
            n_classes=256,
            pred_n_hid=192,
            pred_rnn_layers=1,
            in_feats=40,
            enc_n_hid=176,
            enc_pre_rnn_layers=1,
            enc_post_rnn_layers=1,
            joint_n_hid=192,
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
    ]

    # Multi-model FP8 allocation requires the ProgramBuilder fixes in SDK 29.
    if Version(version("vollo-compiler")) >= Version("29.0.0"):
        models.append(
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
            )
        )

    for x in models:
        yield from _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
