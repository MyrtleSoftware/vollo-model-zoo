# /// script
# requires-python = ">=3.13"
# dependencies = ["beartype", "tabulate", "torch", "vollo-compiler", "vollo-torch"]
# ///
"""
Stateful, two-entrypoint RNN-T model for the Vollo model zoo.

The encoder/joint and prediction/joint entry points share the final joint
network.  The zoo compiles both entry points into one program, matching
the layout used by the ASR runtime.
"""

from collections.abc import Sequence
from contextlib import nullcontext

import torch
import vollo_compiler
import vollo_torch
from beartype import beartype
from tabulate import tabulate
from torch import Tensor, nn


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
        """
        Prediction/joint entry point.

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

        self.scan = vollo_torch.nn.Scan(
            _PredictionJointStep(
                pred_n_hid=pred_n_hid,
                pred_rnn_layers=pred_rnn_layers,
                joint_n_hid=joint_n_hid,
                fp8_weights=fp8_weights,
                joint_network=joint_network,
            )
        )

        self.initial_hidden = nn.Buffer(
            torch.zeros(pred_rnn_layers, 1, pred_n_hid), persistent=False
        )

    def forward(self, embed: Tensor, encoding: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run a sequence of prediction/joint steps.

        Args:
            embed: ``[time, batch, pred_n_hid]`` embedded non-blank tokens.
            encoding: ``[time, batch, joint_n_hid]`` encoder representations.

        Returns:
            A pair ``(prediction, logits)`` of ``[time, batch, joint_n_hid]``
            and ``[time, batch, n_classes]``.
        """
        return self.scan(
            (embed, encoding),
            self.initial_hidden,
            input_axis=(0, 0),
            output_axis=(0, 0),
        )


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
        """
        Encoder/joint entry point.

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

        self.scan = vollo_torch.nn.Scan(
            _EncoderJointStep(
                in_feats=in_feats,
                enc_n_hid=enc_n_hid,
                enc_pre_rnn_layers=enc_pre_rnn_layers,
                enc_post_rnn_layers=enc_post_rnn_layers,
                joint_n_hid=joint_n_hid,
                fp8_weights=fp8_weights,
                joint_network=joint_network,
            )
        )

        self.initial_pre_hidden = nn.Buffer(
            torch.zeros(enc_pre_rnn_layers, 1, enc_n_hid), persistent=False
        )
        self.initial_post_hidden = nn.Buffer(
            torch.zeros(enc_post_rnn_layers, 1, enc_n_hid), persistent=False
        )

    def forward(
        self,
        feats_0: Tensor,
        feats_1: Tensor,
        prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Run a sequence of two-frame encoder/joint steps.

        Args:
            feats_0: ``[time, batch, in_feats]`` first frame of each pair.
            feats_1: ``[time, batch, in_feats]`` second frame of each pair.
            prediction: ``[time, batch, joint_n_hid]`` prediction representations.

        Returns:
            A pair ``(encoding, logits)`` of ``[time, batch, joint_n_hid]`` and
            ``[time, batch, n_classes]``.
        """
        return self.scan(
            (feats_0, feats_1, prediction),
            (self.initial_pre_hidden, self.initial_post_hidden),
            input_axis=(0, 0, 0),
            output_axis=(0, 0),
        )


class _LSTMStack(nn.Module):
    @beartype
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:

        super().__init__()

        self.cells = nn.ModuleList(
            [
                vollo_torch.nn.LSTMCell(
                    input_size if layer == 0 else hidden_size, hidden_size, batch_size=1
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
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
        super().__init__()
        self.fp8_weights = fp8_weights
        self.lstm = _LSTMStack(pred_n_hid, pred_n_hid, pred_rnn_layers)
        self.joint_pred = nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        inputs: Sequence[Tensor],
        hidden: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:

        embed, encoding = inputs

        with vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext():
            prediction, hidden = self.lstm(embed, hidden)
            prediction = self.joint_pred(prediction)
            logits = self.joint_network(encoding + prediction)

        return (prediction, logits), hidden


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
        super().__init__()
        self.fp8_weights = fp8_weights
        self.pre_rnn = _LSTMStack(in_feats, enc_n_hid, enc_pre_rnn_layers)
        self.post_rnn = _LSTMStack(2 * enc_n_hid, enc_n_hid, enc_post_rnn_layers)
        self.joint_enc = nn.Linear(enc_n_hid, joint_n_hid)
        self.joint_network = joint_network

    def forward(
        self,
        inputs: Sequence[Tensor],
        state: Sequence[Tensor],
    ) -> tuple[
        tuple[Tensor, Tensor],
        tuple[Tensor, Tensor],
    ]:
        feats_0, feats_1, prediction = inputs
        pre_hidden, post_hidden = state

        with vollo_torch.Fp8Weights() if self.fp8_weights else nullcontext():
            pre_0, pre_hidden = self.pre_rnn(feats_0, pre_hidden)
            pre_1, pre_hidden = self.pre_rnn(feats_1, pre_hidden)
            output, post_hidden = self.post_rnn(
                torch.cat((pre_0, pre_1), dim=-1),
                post_hidden,
            )
            encoding = self.joint_enc(output)
            logits = self.joint_network(encoding + prediction)

        return (encoding, logits), (pre_hidden, post_hidden)


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
    return [
        dict(
            name="predictor",
            model=prediction,
            inputs=(
                torch.randn(time, batch, pred_n_hid),  # embed
                torch.randn(time, batch, joint_n_hid),  # output from encoder/joint
            ),
            streaming_axis=(0, 0),
            meta={
                "hidden": pred_n_hid,
                "joint_hidden": joint_n_hid,
            },
        ),
        dict(
            name="encoder",
            model=encoder,
            inputs=(
                torch.randn(time, batch, in_feats),  # feats_0
                torch.randn(time, batch, in_feats),  # feats_1
                torch.randn(time, batch, joint_n_hid),  # output from prediction/joint
            ),
            streaming_axis=(0, 0, 0),
            meta={
                "hidden": enc_n_hid,
                "joint_hidden": joint_n_hid,
            },
        ),
    ]


@beartype
def instance(
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
    config: vollo_compiler.Config,
) -> dict:
    return dict(
        entries=multi_model_entries(
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
    )


SIZES = [
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
    # 49M model via fp8 weights, which only some fabrics support.
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
        _features=["fp8"],
    ),
]


if __name__ == "__main__":
    # Standalone: `uv run <this file>` prints each size's VM compute latency on a V80.
    config = vollo_compiler.Config.amd_v80_c6b32()
    print(f"VM estimates on {config.name}:")
    rows = []
    for size in SIZES:
        features = size.get("_features", [])
        if "_experimental" in size or not set(features) <= set(config.features):
            continue
        kwargs = {k: v for k, v in size.items() if not k.startswith("_")}
        args = instance(**kwargs, config=config)
        builder = vollo_compiler.ProgramBuilder(config, remember_allocations=True)
        try:
            for entry in args["entries"]:
                model, _ = vollo_torch.fx.prepare_shape(
                    entry["model"], *entry["inputs"]
                )
                nnir = vollo_torch.fx.nnir.to_nnir(model)
                if entry.get("streaming_axis") is not None:
                    nnir, _ = nnir.streaming_transform(entry["streaming_axis"])
                builder.add_nnir(
                    nnir, entry["name"], quick_compile=args.get("quick_compile", False)
                )
            program = builder.to_program()
            program.pack()
        except (vollo_compiler.AllocationError, vollo_compiler.SaveError) as e:
            program = None
            cycles, latency = f"too big ({type(e).__name__})", None
        # Counted as the zoo does: parameters shared between entries once.
        models = nn.ModuleList(entry["model"] for entry in args["entries"])
        params = sum(p.numel() for p in models.parameters())
        for i, entry in enumerate(args["entries"]):
            if program is not None:
                cycles = program.cycle_count_per_inference(model_index=i)
                latency = program.compute_duration_per_inference_us(
                    model_index=i, spaced=True
                )
            rows.append(
                {
                    "model": entry["name"],
                    "params (M)": params / 1e6,
                    "compute cycles": cycles,
                    "compute latency (us)": latency,
                    **kwargs,
                }
            )
    print(tabulate(rows, headers="keys", floatfmt=".2f"))
