"""Stateful, two-entrypoint RNN-T model for the Vollo model zoo.

The encoder/joint and prediction/joint entry points share the final joint
network.  :func:`main` compiles both entry points into one program, matching
the layout used by the ASR runtime.
"""

from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import Tensor, nn


class _LSTMStack(nn.Module):
    """Apply optimized Vollo LSTM cells while keeping cell state internal."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [
                vollo_torch.nn.LSTMCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    batch_size=1,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Run one time step through every LSTM layer.

        Args:
            x: ``[batch, input_size]`` input to the first layer.
            hidden: ``[num_layers, batch, hidden_size]`` hidden states.

        Returns:
            A pair ``(output, next_hidden)`` where ``output`` is
            ``[batch, hidden_size]`` and ``next_hidden`` is
            ``[num_layers, batch, hidden_size]``.

        The optimized Vollo LSTM cells keep their cell states internally, so
        only the hidden states are carried by the surrounding scan.
        """
        next_hidden = []
        for layer, cell in enumerate(self.cells):
            x = cell(x, hidden[layer])
            next_hidden.append(x)
        return x, torch.stack(next_hidden)


class _PredictionJointStep(nn.Module):
    def __init__(
        self,
        *,
        n_classes: int,
        pred_n_hid: int,
        pred_rnn_layers: int,
        joint_n_hid: int,
        fp8_weights: bool,
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
        self.lstm = _LSTMStack(
            pred_n_hid,
            pred_n_hid,
            pred_rnn_layers,
        )
        self.joint_pred = nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        x: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the prediction network and joint network for one token.

        Args:
            x: ``[batch, (n_classes - 1) + joint_n_hid]``. The first slice is
                a one-hot non-blank token; the second is the current encoder
                representation.
            hidden: ``[pred_rnn_layers, batch, pred_n_hid]`` prediction-network
                hidden states.

        Returns:
            A pair ``(output, next_hidden)``. ``output`` is
            ``[batch, joint_n_hid + n_classes]`` and concatenates the new
            prediction representation with the joint logits. ``next_hidden``
            has the same shape as ``hidden``.
        """
        token = x[..., : self.vocab_without_blank]
        encoding = x[..., self.vocab_without_blank :]
        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            embedded = self.embedding(token)
            prediction, hidden = self.lstm(embedded, hidden)
            prediction = self.joint_pred(prediction)
            logits = self.joint_network(encoding + prediction)
        return torch.cat((prediction, logits), dim=-1), hidden


class PredictionJoint(nn.Module):
    """Run prediction/joint steps with optimized LSTM state on the accelerator."""

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
        self.step = _PredictionJointStep(
            n_classes=n_classes,
            pred_n_hid=pred_n_hid,
            pred_rnn_layers=pred_rnn_layers,
            joint_n_hid=joint_n_hid,
            fp8_weights=fp8_weights,
            joint_network=joint_network,
        )
        self.initial_hidden = nn.Buffer(
            torch.zeros(pred_rnn_layers, 1, pred_n_hid),
            persistent=False,
        )
        self.scan = vollo_torch.nn.Scan(self.step)

    def forward(self, x: Tensor) -> Tensor:
        """Run a sequence of prediction/joint steps.

        Args:
            x: ``[time, batch, (n_classes - 1) + joint_n_hid]``. Each step
                contains a one-hot non-blank token followed by an encoder
                representation.

        Returns:
            ``[time, batch, joint_n_hid + n_classes]``. Each step contains the
            prediction representation followed by the joint logits.
        """
        return self.scan(
            x,
            self.initial_hidden,
            input_axis=0,
            output_axis=0,
        )


class _EncoderJointStep(nn.Module):
    def __init__(
        self,
        *,
        in_feats: int,
        enc_n_hid: int,
        enc_pre_rnn_layers: int,
        enc_post_rnn_layers: int,
        joint_n_hid: int,
        fp8_weights: bool,
        joint_network: nn.Module,
    ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.fp8_weights = fp8_weights
        self.pre_rnn = _LSTMStack(
            in_feats,
            enc_n_hid,
            enc_pre_rnn_layers,
        )
        self.post_rnn = _LSTMStack(
            2 * enc_n_hid,
            enc_n_hid,
            enc_post_rnn_layers,
        )
        self.joint_enc = nn.Linear(enc_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Encode one stacked pair of feature frames and compute joint logits.

        Args:
            x: ``[batch, 2 * in_feats + joint_n_hid]``. The first two slices
                are consecutive acoustic feature frames; the final slice is
                the current prediction representation.
            state: ``(pre_hidden, post_hidden)``. Their shapes are
                ``[enc_pre_rnn_layers, batch, enc_n_hid]`` and
                ``[enc_post_rnn_layers, batch, enc_n_hid]``.

        Returns:
            A pair ``(output, next_state)``. ``output`` is
            ``[batch, joint_n_hid + n_classes]`` and concatenates the encoder
            representation with the joint logits. ``next_state`` contains the
            updated pre-RNN and post-RNN hidden states.
        """
        features = x[..., : 2 * self.in_feats]
        prediction = x[..., 2 * self.in_feats :]
        pre_hidden, post_hidden = state
        feature_0 = features[:, : self.in_feats]
        feature_1 = features[:, self.in_feats :]

        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            pre_0, pre_hidden = self.pre_rnn(feature_0, pre_hidden)
            pre_1, pre_hidden = self.pre_rnn(feature_1, pre_hidden)
            output, post_hidden = self.post_rnn(
                torch.cat((pre_0, pre_1), dim=-1),
                post_hidden,
            )
            encoding = self.joint_enc(output)
            logits = self.joint_network(encoding + prediction)

        return torch.cat((encoding, logits), dim=-1), (pre_hidden, post_hidden)


class EncoderJoint(nn.Module):
    """Run encoder/joint steps with optimized LSTM state on the accelerator."""

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
        self.step = _EncoderJointStep(
            in_feats=in_feats,
            enc_n_hid=enc_n_hid,
            enc_pre_rnn_layers=enc_pre_rnn_layers,
            enc_post_rnn_layers=enc_post_rnn_layers,
            joint_n_hid=joint_n_hid,
            fp8_weights=fp8_weights,
            joint_network=joint_network,
        )
        self.initial_pre_hidden = nn.Buffer(
            torch.zeros(enc_pre_rnn_layers, 1, enc_n_hid),
            persistent=False,
        )
        self.initial_post_hidden = nn.Buffer(
            torch.zeros(enc_post_rnn_layers, 1, enc_n_hid),
            persistent=False,
        )
        self.scan = vollo_torch.nn.Scan(self.step)

    def forward(self, x: Tensor) -> Tensor:
        """Run a sequence of two-frame encoder/joint steps.

        Args:
            x: ``[time, batch, 2 * in_feats + joint_n_hid]``. Each step
                contains two acoustic feature frames followed by a prediction
                representation.

        Returns:
            ``[time, batch, joint_n_hid + n_classes]``. Each step contains the
            encoder representation followed by the joint logits.
        """
        return self.scan(
            x,
            (self.initial_pre_hidden, self.initial_post_hidden),
            input_axis=0,
            output_axis=0,
        )


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
    """Compile the internal-state prediction and encoder entry points."""
    from vollo_model_zoo.vm import MultiModelEntry, vollo_multi_model_info

    joint_network = nn.Sequential(
        nn.ReLU(),
        nn.Linear(joint_n_hid, n_classes),
    )
    prediction = PredictionJoint(
        n_classes=n_classes,
        pred_n_hid=pred_n_hid,
        pred_rnn_layers=pred_rnn_layers,
        joint_n_hid=joint_n_hid,
        fp8_weights=fp8_weights,
        joint_network=joint_network,
    )
    encoder = EncoderJoint(
        in_feats=in_feats,
        enc_n_hid=enc_n_hid,
        enc_pre_rnn_layers=enc_pre_rnn_layers,
        enc_post_rnn_layers=enc_post_rnn_layers,
        joint_n_hid=joint_n_hid,
        fp8_weights=fp8_weights,
        joint_network=joint_network,
    )
    prediction_input = torch.randn(3, 1, n_classes - 1 + joint_n_hid)
    # paired-frame input for encoder: 2 * in_feats
    encoder_input = torch.randn(3, 1, 2 * in_feats + joint_n_hid)
    return vollo_multi_model_info(
        [
            MultiModelEntry(
                name="prediction_joint",
                model=prediction,
                inputs=(prediction_input,),
                streaming_axis=0,
            ),
            MultiModelEntry(
                name="encoder_joint",
                model=encoder,
                inputs=(encoder_input,),
                streaming_axis=0,
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
        # 49M model with FP8 weights.
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
