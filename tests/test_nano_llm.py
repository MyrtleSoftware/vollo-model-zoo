"""
Equivalence tests for `nano-llm.py`, against the architecture it is a port of.

NanoLLM is [nanochat](https://github.com/karpathy/nanochat) with
each block's causal self-attention is replaced by a FLA Mamba-2 mixer.

The tests below `convert_state_dict` run against a synthetic reference module,
so they run anywhere. The checkpoint tests at the bottom load Myrtle's trained
140M-parameter NanoLLM off an internal mount and greedy-decode from it; they
skip when the mount isn't present
"""

import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from fla.layers.mamba2 import Mamba2 as FLAMamba2
from test_mamba2 import convert_state_dict as convert_mixer_state_dict
from torch import nn

NanoLLM = importlib.import_module("vollo_model_zoo.models.nano-llm").NanoLLM

D_MODEL = 64
D_STATE = 16
D_CONV = 4
D_HEAD = 16
EXPAND = 2
MLP_DIM = 128
N_LAYERS = 2
N_HEADS = (D_MODEL * EXPAND) // D_HEAD
VOCAB_SIZE = 100
PARTITIONS = [(None, False), (1, True), (4, False), (4, True)]


@beartype
def convert_state_dict(state_dict: dict, vocab_size: int | None = None) -> dict:
    """
    Convert a trained nanochat-style checkpoint to Vollo NanoLLM state dict.
    """
    out = {}

    for dst, src in (
        ("inp_embeddings.weight", "transformer.wte.weight"),
        ("head.weight", "lm_head.weight"),
    ):
        weight = state_dict[src]
        if vocab_size is not None:
            assert vocab_size <= weight.shape[0], (
                f"{src} has {weight.shape[0]} rows, fewer than the"
                f" {vocab_size}-token vocabulary it is meant to cover"
            )
            weight = weight[:vocab_size]
        out[dst] = weight

    n_layers = 1 + max(
        int(k.split(".")[2]) for k in state_dict if k.startswith("transformer.h.")
    )

    for i in range(n_layers):
        src = f"transformer.h.{i}."

        mixer = src + "attn.mamba2."
        mixer_state_dict = {
            k.removeprefix(mixer): v
            for k, v in state_dict.items()
            if k.startswith(mixer)
        }
        for k, v in convert_mixer_state_dict(mixer_state_dict).items():
            out[f"blocks.{i}.mixer.{k}"] = v

        out[f"blocks.{i}.mlp.ffn1.weight"] = state_dict[src + "mlp.c_fc.weight"]
        out[f"blocks.{i}.mlp.ffn2.weight"] = state_dict[src + "mlp.c_proj.weight"]

    return out


def _norm() -> nn.RMSNorm:
    return nn.RMSNorm(D_MODEL, eps=1e-5, elementwise_affine=False)


class _RefMLP(nn.Module):

    @beartype
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(D_MODEL, MLP_DIM, bias=False)
        self.c_proj = nn.Linear(MLP_DIM, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class _RefAttn(nn.Module):

    @beartype
    def __init__(self):
        super().__init__()
        self.mamba2 = FLAMamba2(
            num_heads=N_HEADS,
            head_dim=D_HEAD,
            hidden_size=D_MODEL,
            state_size=D_STATE,
            expand=EXPAND,
            conv_kernel=D_CONV,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            rms_norm=True,
            backend="triton",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mamba2(x)
        return out[0] if isinstance(out, tuple) else out


class _RefBlock(nn.Module):

    @beartype
    def __init__(self):
        super().__init__()
        self.attn = _RefAttn()
        self.mlp = _RefMLP()
        self.norm = _norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x


class _RefNanochat(nn.Module):
    """
    Nanochat's GPT with FLA's Mamba2 in place of attention
    """

    @beartype
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(VOCAB_SIZE, D_MODEL),
                h=nn.ModuleList(_RefBlock() for _ in range(N_LAYERS)),
            )
        )
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.norm = _norm()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Input:
            idx: [B, T]
        Output:
            logits: [B, T, vocab_size]
        """
        x = self.norm(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.norm(x)
        softcap = 15
        return softcap * torch.tanh(self.lm_head(x) / softcap)


@beartype
def _nano_llm(head_partitions: int | None, distributed_norm: bool) -> nn.Module:
    return NanoLLM(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        d_head=D_HEAD,
        expand=float(EXPAND),
        mlp_dim=MLP_DIM,
        n_layers=N_LAYERS,
        vocab_size=VOCAB_SIZE,
        head_partitions=head_partitions,
        distributed_norm=distributed_norm,
    ).eval()


@pytest.mark.parametrize("head_partitions, distributed_norm", PARTITIONS)
def test_nano_llm_state_dict_conversion(head_partitions, distributed_norm: bool):
    torch.manual_seed(42)

    converted = convert_state_dict(_RefNanochat().state_dict())

    # strict: every parameter the model needs, at the right shape, and nothing
    # it does not. The mixer's own load hook re-splits the wide keys the
    # conversion emits into whichever partition layout this model holds.
    _nano_llm(head_partitions, distributed_norm).load_state_dict(converted, strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for triton")
@pytest.mark.parametrize("head_partitions, distributed_norm", PARTITIONS)
def test_nano_llm_equivalence(head_partitions, distributed_norm: bool):
    torch.manual_seed(42)

    ref = _RefNanochat().eval().cuda()

    model = _nano_llm(head_partitions, distributed_norm).cuda()
    model.load_state_dict(convert_state_dict(ref.state_dict()), strict=True)

    T = 32
    tokens = torch.randint(VOCAB_SIZE, (T,)).cuda()

    with torch.no_grad():
        ref_out = ref(tokens.unsqueeze(0)).squeeze(0)
        out = model(model.embed(tokens))

    assert ref_out.shape == out.shape

    for t in range(T):
        np.testing.assert_allclose(
            ref_out[t].cpu().numpy(),
            out[t].cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"NanoLLM output mismatch at step={t}",
        )


# Trained-checkpoint tests.

CHECKPOINTS = Path("/mount/cassini/checkpoints/nanollm")
CHECKPOINT = CHECKPOINTS / "mamba_140M"
TOKENIZER = CHECKPOINTS / "tokenizer" / "tokenizer.pkl"

# Greedy decode is deterministic, so a trained model has exactly one
# continuation for a prompt.
PROMPT = "The capital of France is"
CONTINUATION = " Paris"

# (head_partitions, distributed_norm)
CHECKPOINT_PARTITIONS = [(None, False), (24, True), (24, False)]

requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT.is_dir() and TOKENIZER.is_file()),
    reason=f"No trained NanoLLM checkpoint at {CHECKPOINT}",
)


@beartype
def _newest(directory: Path, pattern: str) -> Path:
    """
    The highest-numbered `<prefix>_<step>.<ext>` in `directory`.

    Checkpoints are written per training step, so this picks up a retrain
    without the test naming a step.
    """
    matches = sorted(directory.glob(pattern))
    assert matches, f"No {pattern} in {directory}"
    return matches[-1]


@pytest.fixture(scope="session")
@beartype
def checkpoint() -> tuple[dict, dict]:
    """
    `(model_config, vollo_state_dict)` for the trained NanoLLM.
    """
    config = json.loads(_newest(CHECKPOINT, "meta_*.json").read_text())["model_config"]
    state_dict = torch.load(
        _newest(CHECKPOINT, "model_*.pt"), map_location="cpu", weights_only=True
    )
    return config, convert_state_dict(state_dict, vocab_size=config["vocab_size"])


@pytest.fixture(scope="session")
@beartype
def tokenizer():
    """
    The `tiktoken` encoding the checkpoint was trained against.

    Pickled by nanochat rather than saved in a tiktoken-native format, so it
    needs `tiktoken` importable to unpickle -- it reconstructs a
    `tiktoken.core.Encoding`.
    """
    pytest.importorskip("tiktoken")
    with open(TOKENIZER, "rb") as f:
        return pickle.load(f)


@beartype
def _from_checkpoint(
    checkpoint: tuple[dict, dict],
    head_partitions: int | None,
    distributed_norm: bool,
) -> nn.Module:
    """
    Build the Vollo NanoLLM model from the checkpoint and load the weights into it.
    """
    config, state_dict = checkpoint

    model = NanoLLM(
        d_model=config["n_embd"],
        d_state=config["state_size"],
        d_conv=config["conv_kernel"],
        d_head=config["head_dim"],
        expand=float(config["expand"]),
        mlp_dim=state_dict["blocks.0.mlp.ffn1.weight"].shape[0],
        n_layers=config["n_layer"],
        vocab_size=config["vocab_size"],
        head_partitions=head_partitions,
        distributed_norm=distributed_norm,
    ).eval()
    model.load_state_dict(state_dict, strict=True)
    return model


@beartype
def _greedy_decode(model: nn.Module, prompt_ids: list[int], n: int) -> list[int]:
    """
    Greedily decode `n` tokens, returning just the generated ids.

    Note that for the test, the whole prefix is re-run per token since
    there is no state to carry between calls.
    """
    tokens = list(prompt_ids)

    with torch.no_grad():
        for _ in range(n):
            logits = model(model.embed(torch.tensor(tokens)))
            tokens.append(int(logits[-1].argmax()))

    return tokens[len(prompt_ids) :]


@requires_checkpoint
def test_checkpoint_decode(checkpoint, tokenizer):
    """
    Test that the trained model, loaded through `convert_state_dict`
    decodes sensibly.

    This is the end-to-end demo the README points at: host-side tokenize and
    embed, then the compiled model's `forward`, which returns capped logits.
    """
    model = _from_checkpoint(checkpoint, head_partitions=24, distributed_norm=True)

    prompt_ids = tokenizer.encode_ordinary(PROMPT)
    generated = tokenizer.decode(_greedy_decode(model, prompt_ids, n=16))

    print(f"\n{PROMPT!r} ->{generated!r}")
    assert generated.startswith(CONTINUATION), (
        f"Expected the trained model to continue {PROMPT!r} with"
        f" {CONTINUATION!r}, got {generated!r}"
    )
