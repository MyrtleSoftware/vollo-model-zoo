from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn

from vollo_model_zoo.models.mamba2 import Mamba2


def _norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-5, elementwise_affine=False)


class _MLPBlock(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ffn_fp8: bool,
    ):
        super().__init__()
        self.ffn1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.ffn2 = nn.Linear(hidden_dim, dim, bias=False)
        self.fp8_context = vollo_torch.Fp8Weights if ffn_fp8 else nullcontext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self.fp8_context():
            x = self.ffn1(x)
            x = self.act(x)
            x = x * x
            x = self.ffn2(x)
        return x


class _Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        d_head,
        expand,
        mlp_dim,
        head_partitions,
        distributed_norm,
        ssm_fp32=True,
        ffn_fp8=True,
    ):
        super().__init__()
        self.mixer = Mamba2(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=d_head,
            expand=expand,
            head_partitions=head_partitions,
            distributed_norm=distributed_norm,
            ssm_fp32=ssm_fp32,
        )
        self.mlp = _MLPBlock(d_model, mlp_dim, ffn_fp8)
        self.norm = _norm(d_model)

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x


class Mamba2LM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        d_head,
        expand,
        mlp_dim,
        head_partitions,
        distributed_norm,
        n_layers,
        vocab_size,
        pad_vocab_size_to=64,
        ssm_fp32=True,
        ffn_fp8=True,
    ):
        super().__init__()
        # The vocab is padded to a multiple of `pad_vocab_size_to` so the head is
        # a clean shape, and `get_logits` slices the padding back off the logits.
        padded_vocab_size = (
            (vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        self.vocab_size = vocab_size
        self.inp_embeddings = nn.Embedding(padded_vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[
                _Block(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    d_head=d_head,
                    expand=expand,
                    mlp_dim=mlp_dim,
                    head_partitions=head_partitions,
                    distributed_norm=distributed_norm,
                    ssm_fp32=ssm_fp32,
                    ffn_fp8=ffn_fp8,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, padded_vocab_size, bias=False)
        self.norm = _norm(d_model)
        self.fp8_context = vollo_torch.Fp8Weights if ffn_fp8 else nullcontext

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Token ids [T] -> embeddings [T, d_model].

        """
        return self.inp_embeddings(tokens)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., : self.vocab_size]
        softcap = 15
        logits = softcap * torch.tanh(x / softcap)
        return logits

    def forward(self, x):
        """
        Input:
            x: [T, d_model]  -- embeddings, from `embed`
        Output:
            logits: [T, vocab_size]
        """
        x = self.norm(x)
        x = self.norm(self.blocks(x))
        with self.fp8_context():
            x = self.head(x)
        return x


@beartype
def convert_state_dict(state_dict: dict) -> dict:
    """
    Convert a nanochat mamba2 checkpoint to a Mamba2LM state dict.
    """
    out = {}
    out["inp_embeddings.weight"] = state_dict["transformer.wte.weight"].float()
    out["head.weight"] = state_dict["lm_head.weight"].float()

    n_layers = 1 + max(
        int(k.split(".")[2]) for k in state_dict if k.startswith("transformer.h.")
    )

    for i in range(n_layers):
        src = f"transformer.h.{i}."
        mix, dst = src + "attn.mamba2.", f"blocks.{i}.mixer."

        # A_log is [n_heads]; out_proj is [d_model, d_inner].
        n_heads = state_dict[mix + "A_log"].shape[0]
        d_inner = state_dict[mix + "out_proj.weight"].shape[1]
        d_head = d_inner // n_heads
        # in_proj is [2 * d_inner + 2 * d_state + n_heads, d_model]
        d_state = (
            state_dict[mix + "in_proj.weight"].shape[0] - 2 * d_inner - n_heads
        ) // 2

        z, x, B, C, dt = torch.split(
            state_dict[mix + "in_proj.weight"].float(),
            [d_inner, d_inner, d_state, d_state, n_heads],
            dim=0,
        )
        out[dst + "proj_z.weight"] = z
        out[dst + "proj_x.weight"] = x
        out[dst + "proj_B.weight"] = B
        out[dst + "proj_C.weight"] = C
        out[dst + "proj_dt.weight"] = dt

        splits = [d_inner, d_state, d_state]
        weights = torch.split(state_dict[mix + "conv1d.weight"].float(), splits, dim=0)
        biases = torch.split(state_dict[mix + "conv1d.bias"].float(), splits, dim=0)
        for name, w, b in zip(("conv_x", "conv_B", "conv_C"), weights, biases):
            out[f"{dst}{name}.conv1d.conv.weight"] = w
            out[f"{dst}{name}.conv1d.conv.bias"] = b

        out[dst + "dt_bias"] = state_dict[mix + "dt_bias"].float()
        out[dst + "A_log"] = state_dict[mix + "A_log"].float()
        # [n_heads] -> [d_inner]: upstream broadcasts D over each head's features.
        out[dst + "D"] = state_dict[mix + "D"].float().repeat_interleave(d_head)
        out[dst + "norm.weight"] = state_dict[mix + "norm.weight"].float()
        out[dst + "out_proj.weight"] = state_dict[mix + "out_proj.weight"].float()

        out[f"blocks.{i}.mlp.ffn1.weight"] = state_dict[src + "mlp.c_fc.weight"].float()
        out[f"blocks.{i}.mlp.ffn2.weight"] = state_dict[
            src + "mlp.c_proj.weight"
        ].float()

    return out


@beartype
def _vm(
    d_model: int,
    d_state: int,
    d_conv: int,
    d_head: int,
    expand: float,
    n_layers: int,
    vocab_size: int,
    config: str,
    head_partitions: int | None = None,
    distributed_norm: bool = False,
    ssm_fp32: bool = True,
    ffn_fp8: bool = True,
):
    # Defer import
    from vollo_model_zoo.vm import vollo_info

    input = torch.randn(5, d_model)

    model = Mamba2LM(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        d_head=d_head,
        expand=expand,
        mlp_dim=4 * d_model,
        head_partitions=head_partitions,
        distributed_norm=distributed_norm,
        n_layers=n_layers,
        vocab_size=vocab_size,
        ssm_fp32=ssm_fp32,
        ffn_fp8=ffn_fp8,
    )

    return vollo_info(
        model,
        input,
        config=config,
        time_axis=0,
        allow_dynamic_weights=True,
        quick_compile=True,
        allow_unserializable=True,
        meta=dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=d_head,
            expand=expand,
            mlp_dim=4 * d_model,
            head_partitions=head_partitions,
            distributed_norm=distributed_norm,
            n_layers=n_layers,
            vocab_size=vocab_size,
        ),
    )


@beartype
def main(config: str = "V80plus") -> Generator:
    for x in [
        # dict(d_model=384, d_state=32, d_conv=4, d_head=32, expand=2.0, head_partitions=24, n_layers=1, vocab_size=1024, ssm_fp32=False, ffn_fp8=True),
        # dict(d_model=256, d_state=32, d_conv=4, d_head=32, expand=2, head_partitions=None, n_layers=1, vocab_size=1024, ssm_fp32=False),
        dict(
            d_model=768,
            d_state=64,
            d_conv=4,
            d_head=48,
            expand=1.5,
            head_partitions=24,
            distributed_norm=False,
            n_layers=12,
            vocab_size=32768,
            ssm_fp32=False,
            ffn_fp8=True,
        ),
        # dict(
        #     d_model=768,
        #     d_state=64,
        #     d_conv=4,
        #     d_head=64,
        #     expand=2,
        #     head_partitions=None,
        #     n_layers=12,
        #     vocab_size=32768,
        #     ssm_fp32=True,
        #     ffn_fp8=True,
        # ),
        # 151.1M params max
        # dict(
        #     d_model=768,
        #     d_state=64,
        #     d_conv=4,
        #     d_head=64,
        #     expand=2,
        #     head_partitions=None,
        #     n_layers=12,
        #     vocab_size=32768,
        #     ssm_fp32=False,
        #     ffn_fp8=True,
        # ),
        # 63.6M params
        # dict(
        #     d_model=512,
        #     d_state=64,
        #     d_conv=4,
        #     d_head=64,
        #     expand=2,
        #     head_partitions=None,
        #     n_layers=8,
        #     vocab_size=32768,
        #     ssm_fp32=True,
        #     ffn_fp8=False,
        # ),
        # this works: this is the max with 24 cores (101.2M)
        # dict(
        #     d_model=640,
        #     d_state=128,
        #     d_conv=4,
        #     d_head=64,
        #     expand=2,
        #     head_partitions=None,
        #     n_layers=10,
        #     vocab_size=32768,
        #     ssm_fp32=False,
        #     ffn_fp8=False,
        # ),
    ]:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
