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
    @beartype
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        d_head: int,
        expand: float,
        mlp_dim: int,
        head_partitions: int | None = None,
        distributed_norm: bool = False,
        ssm_fp32: bool = True,
        ffn_fp8: bool = False,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x


class NanoLLM(nn.Module):
    @beartype
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        d_head: int,
        expand: float,
        mlp_dim: int,
        n_layers: int,
        vocab_size: int,
        head_partitions: int | None = None,
        distributed_norm: bool = False,
        ssm_fp32: bool = True,
        ffn_fp8: bool = False,
    ):
        super().__init__()

        self.inp_embeddings = nn.Embedding(vocab_size, d_model)
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.norm = _norm(d_model)
        self.fp8_context = vollo_torch.Fp8Weights if ffn_fp8 else nullcontext

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Token ids [T] -> embeddings [T, d_model].

        """
        return self.inp_embeddings(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [T, d_model]
        Output:
            logits: [T, vocab_size]
        """
        x = self.norm(x)
        x = self.norm(self.blocks(x))
        with self.fp8_context():
            x = self.head(x)
        # Softcapping inherited from the original nanochat repo
        softcap = 15
        return softcap * torch.tanh(x / softcap)


@beartype
def _vm(
    d_model: int,
    d_state: int,
    d_conv: int,
    d_head: int,
    expand: float,
    mlp_dim: int,
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

    model = NanoLLM(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        d_head=d_head,
        expand=expand,
        mlp_dim=mlp_dim,
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
            d_head=d_head,
            expand=expand,
            n_layers=n_layers,
        ),
    )


@beartype
def main(config: str = "4xV80") -> Generator:
    from vollo_model_zoo.vm import config_supports

    models = [
        # ~1M parameter baseline
        dict(
            d_model=256,
            d_state=32,
            d_conv=4,
            d_head=32,
            expand=2.0,
            mlp_dim=640,
            head_partitions=None,
            n_layers=1,
            vocab_size=512,
            ssm_fp32=False,
            ffn_fp8=False,
        ),
        dict(
            d_model=384,
            d_state=32,
            d_conv=4,
            d_head=32,
            expand=2.0,
            mlp_dim=1536,
            head_partitions=None,
            n_layers=1,
            vocab_size=1024,
            ssm_fp32=False,
            ffn_fp8=False,
        ),
        # 140M model Myrtle trained; see README.md
        dict(
            d_model=768,
            d_state=64,
            d_conv=4,
            d_head=48,
            expand=1.5,
            mlp_dim=3072,
            head_partitions=24,
            n_layers=12,
            vocab_size=32768,
            ssm_fp32=False,
            ffn_fp8=True,
        ),
    ]

    for x in models:
        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
