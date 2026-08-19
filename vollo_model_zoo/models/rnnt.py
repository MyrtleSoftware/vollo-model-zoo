"""Stateful, two-entrypoint RNN-T model for the Vollo model zoo.

The encoder/joint and prediction/joint entry points share the final joint
network.  :func:`main` compiles both entry points into one program, matching
the layout used by the ASR runtime.
"""

from collections.abc import Generator, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import Tensor, nn


class _LSTMStack(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        """Apply optimized Vollo LSTM cells while keeping cell state internal.
        
        Args:
               input_size:           Features in the input to the first layer
               hidden_size:          Features in each layer's hidden state
               num_layers:           Number of stacked LSTM layers
        """
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
        """
        next_hidden = []
        for layer, cell in enumerate(self.cells):
            x = cell(x, hidden[layer])
            next_hidden.append(x)
        return x, torch.stack(next_hidden)


class _PredictionJointStep(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        pred_n_hid: int,
        pred_rnn_layers: int,
        joint_n_hid: int,
        fp8_weights: bool,
        joint_network: nn.Module,
    ) -> None:
        """One prediction/joint step, for use as a :class:`Scan` step.

        Args:
               pred_n_hid:           Width of the prediction network's LSTM
               pred_rnn_layers:      Number of prediction LSTM layers
               joint_n_hid:          Width of the shared joint space
               fp8_weights:          Store weight matrices as FP8 E4M3
               joint_network:        Joint network, shared with the encoder
        """
        super().__init__()

        self.fp8_weights = fp8_weights
        self.lstm = _LSTMStack(pred_n_hid, pred_n_hid, pred_rnn_layers)
        self.joint_pred = nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        inputs: Sequence[Tensor],  # (embed, encoding)
        hidden: Tensor,  # [pred_rnn_layers, batch, pred_n_hid]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:  # (prediction, logits), hidden
        """Run the prediction network and joint network for one token.

        Args:
            inputs: ``(embed, encoding)``. ``embed`` is
                ``[batch, pred_n_hid]``, the host's embedding look-up for the
                latest non-blank token. ``encoding`` is ``[batch, joint_n_hid]``,
                the current encoder representation.
            hidden: ``[pred_rnn_layers, batch, pred_n_hid]`` prediction-network
                hidden states.

        Returns:
            A pair ``((prediction, logits), next_hidden)``, where ``prediction``
            is ``[batch, joint_n_hid]``, ``logits`` is ``[batch, n_classes]``
            and ``next_hidden`` matches ``hidden``.
        """
        embed, encoding = inputs

        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            prediction, hidden = self.lstm(embed, hidden)
            prediction = self.joint_pred(prediction)
            logits = self.joint_network(encoding + prediction)

        return (prediction, logits), hidden


class PredictionJoint(nn.Module):
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
        """Prediction/joint entry point.

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

        self.step = _PredictionJointStep(
            pred_n_hid=pred_n_hid,
            pred_rnn_layers=pred_rnn_layers,
            joint_n_hid=joint_n_hid,
            fp8_weights=fp8_weights,
            joint_network=joint_network,
        )
        self.scan = vollo_torch.nn.Scan(self.step)

        self.initial_hidden = nn.Buffer(
            torch.zeros(pred_rnn_layers, 1, pred_n_hid),
            persistent=False,
        )

    def forward(self, embed: Tensor, encoding: Tensor) -> tuple[Tensor, Tensor]:
        """Run a sequence of prediction/joint steps.

        Args:
            embed: ``[time, batch, pred_n_hid]`` embedded non-blank tokens.
            encoding: ``[time, batch, joint_n_hid]`` encoder representations.

        Returns:
            A pair ``(prediction, logits)`` of ``[time, batch, joint_n_hid]``
            and ``[time, batch, n_classes]``.
        """
        outputs = self.scan(
            (embed, encoding),
            self.initial_hidden,
            input_axis=(0, 0),
            output_axis=(0, 0),
        )
        return outputs[0], outputs[1]


class _EncoderJointStep(nn.Module):
    @beartype
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
        """One two-frame encoder/joint step, for use as a :class:`Scan` step.

        Args:
               in_feats:             Acoustic features per frame
               enc_n_hid:            Width of the encoder's LSTMs
               enc_pre_rnn_layers:   Pre-RNN layers, advanced once per frame
               enc_post_rnn_layers:  Post-RNN layers, advanced once per pair
               joint_n_hid:          Width of the shared joint space
               fp8_weights:          Store weight matrices as FP8 E4M3
               joint_network:        Joint network, shared with the predictor
        """
        super().__init__()

        self.fp8_weights = fp8_weights
        self.pre_rnn = _LSTMStack(in_feats, enc_n_hid, enc_pre_rnn_layers)
        self.post_rnn = _LSTMStack(2 * enc_n_hid, enc_n_hid, enc_post_rnn_layers)
        self.joint_enc = nn.Linear(enc_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        inputs: Sequence[Tensor],  # (feats_0, feats_1, prediction)
        state: Sequence[Tensor],  # (pre_hidden, post_hidden)
    ) -> tuple[
        tuple[Tensor, Tensor],  # (encoding, logits)
        tuple[Tensor, Tensor],  # (pre_hidden, post_hidden)
    ]:
        """Encode a pair of feature frames and compute joint logits.

        Taking two frames per step keeps the accelerator interface fixed-rate:
        the pre-RNN advances twice, then the post-RNN advances once per pair.

        Args:
            inputs: ``(feats_0, feats_1, prediction)``. The two frames are
                ``[batch, in_feats]`` each; ``prediction`` is
                ``[batch, joint_n_hid]``.
            state: ``(pre_hidden, post_hidden)``, shaped
                ``[enc_pre_rnn_layers, batch, enc_n_hid]`` and
                ``[enc_post_rnn_layers, batch, enc_n_hid]``.

        Returns:
            A pair ``((encoding, logits), (pre_hidden, post_hidden))``, where
            ``encoding`` is ``[batch, joint_n_hid]`` and ``logits`` is
            ``[batch, n_classes]``.
        """
        feats_0, feats_1, prediction = inputs
        pre_hidden, post_hidden = state

        precision = vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext()
        with precision:
            pre_0, pre_hidden = self.pre_rnn(feats_0, pre_hidden)
            pre_1, pre_hidden = self.pre_rnn(feats_1, pre_hidden)
            output, post_hidden = self.post_rnn(
                torch.cat((pre_0, pre_1), dim=-1),
                post_hidden,
            )
            encoding = self.joint_enc(output)
            logits = self.joint_network(encoding + prediction)

        return (encoding, logits), (pre_hidden, post_hidden)


class EncoderJoint(nn.Module):
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
        """Encoder/joint entry point.

        Args:
               in_feats:             Acoustic features per frame (two per step)
               enc_n_hid:            Width of the encoder's LSTMs
               enc_pre_rnn_layers:   Pre-RNN layers, advanced once per frame
               enc_post_rnn_layers:  Post-RNN layers, advanced once per pair
               joint_n_hid:          Width of the shared joint space
               joint_network:        Joint network, shared with the predictor so
                                     the compiler packs its weights once
               fp8_weights:          Store weight matrices as FP8 E4M3
        """
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
        self.scan = vollo_torch.nn.Scan(self.step)

        self.initial_pre_hidden = nn.Buffer(
            torch.zeros(enc_pre_rnn_layers, 1, enc_n_hid),
            persistent=False,
        )
        self.initial_post_hidden = nn.Buffer(
            torch.zeros(enc_post_rnn_layers, 1, enc_n_hid),
            persistent=False,
        )

    def forward(
        self,
        feats_0: Tensor,
        feats_1: Tensor,
        prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run a sequence of two-frame encoder/joint steps.

        Args:
            feats_0: ``[time, batch, in_feats]`` first frame of each pair.
            feats_1: ``[time, batch, in_feats]`` second frame of each pair.
            prediction: ``[time, batch, joint_n_hid]`` prediction representations.

        Returns:
            A pair ``(encoding, logits)`` of ``[time, batch, joint_n_hid]`` and
            ``[time, batch, n_classes]``.
        """
        outputs = self.scan(
            (feats_0, feats_1, prediction),
            (self.initial_pre_hidden, self.initial_post_hidden),
            input_axis=(0, 0, 0),
            output_axis=(0, 0),
        )
        return outputs[0], outputs[1]


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

    time, batch = 3, 1
    return vollo_multi_model_info(
        [
            MultiModelEntry(
                name="prediction_joint",
                model=prediction,
                inputs=(
                    torch.randn(time, batch, pred_n_hid),  # embed
                    torch.randn(time, batch, joint_n_hid),  # output from encoder/joint
                ),
                streaming_axis=(0, 0),
            ),
            MultiModelEntry(
                name="encoder_joint",
                model=encoder,
                inputs=(
                    torch.randn(time, batch, in_feats),  # feats_0
                    torch.randn(time, batch, in_feats),  # feats_1
                    torch.randn(time, batch, joint_n_hid),  # output from prediction/joint
                ),
                streaming_axis=(0, 0, 0),
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
