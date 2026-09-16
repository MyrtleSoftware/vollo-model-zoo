from collections.abc import Generator
from contextlib import nullcontext
from math import ceil
from pathlib import Path

import torch
import vollo_torch
from beartype import beartype
from torch import nn

from vollo_model_zoo.models.mamba2 import Mamba2


def _norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-5, elementwise_affine=False)


@beartype
def head_partitions_for(n_heads: int, num_cores: int) -> int | None:
    """
    Pick `head_partitions` for `n_heads` heads for a config of `num_cores` cores.

    `head_partitions` is the number of head groups the model's heads are split
    into, and group `p` runs on core `p`, so there are at most `num_cores` of
    them, and since the cores run in parallel the latency depends on whatever the
    largest group costs. This returns the fewest partitions that still
    reach the smallest achievable largest group, and `None` if that is one group.

    e.g. 16 heads on 24 cores gives 16 (one head each, 8 cores unused); on 6
    cores it gives 6 (groups of 3,3,3,3,2,2); 7 heads on 6 cores gives 4
    (2,2,2,1) rather than 6 (2,1,1,1,1,1), which is no faster. `None` when there
    is nothing to split across: 16 heads on a single-core config, or a one-head
    model on any config.
    """
    limit = min(n_heads, num_cores)
    largest_group = ceil(n_heads / limit)  # the smallest max group size we can reach
    partitions = ceil(n_heads / largest_group)  # fewest groups of at most that size
    return partitions if partitions > 1 else None


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
    distributed_norm: bool = False,
    ssm_fp32: bool = True,
    ffn_fp8: bool = True,
):
    # Defer import
    from vollo_model_zoo.vm import CONFIGS, vollo_info

    # Split the model's heads across the cores of whichever config we compile for.
    n_heads = int(expand * d_model) // d_head
    head_partitions = head_partitions_for(n_heads, CONFIGS[config].num_cores)

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
            head_partitions=head_partitions,
            ffn_fp8=ffn_fp8,
        ),
    )


@beartype
def main(config: str = "4xV80") -> Generator:
    from vollo_model_zoo.vm import CONFIGS

    models = [
        # ~1M parameter baseline
        dict(
            d_model=256,
            d_state=32,
            d_conv=4,
            d_head=32,
            expand=1.5,
            mlp_dim=768,
            n_layers=1,
            vocab_size=512,
            ssm_fp32=True,
            ffn_fp8=False,
        ),
        dict(
            d_model=384,
            d_state=32,
            d_conv=4,
            d_head=32,
            expand=2.0,
            mlp_dim=1536,
            n_layers=1,
            vocab_size=1024,
            ssm_fp32=False,
            ffn_fp8=False,
        ),
        # 140M model Myrtle trained; see README.md. Its weights need four V80s
        # worth of weight store, so it is only reachable on a config with the
        # cores to hold it — a smaller board reports it as an `AllocationError`.
        dict(
            d_model=768,
            d_state=64,
            d_conv=4,
            d_head=48,
            expand=1.5,
            mlp_dim=3072,
            n_layers=12,
            vocab_size=32768,
            ssm_fp32=False,
            ffn_fp8=True,
            min_cores=24,
        ),
    ]

    for size in models:
        x = dict(size)

        # Leave out the 140M size the requires the experimental config to run
        if CONFIGS[config].num_cores < x.pop("min_cores", 1):
            continue

        yield _vm(**x, config=config)


if __name__ == "__main__":
    print(f"Model '{Path(__file__).stem}':")
    for result in main():
        print(f"\t{result}")
