from functools import cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_MANTISSA = 23
_EXP_SIGN = 9


@cache
def _round_impl(n: int, trunc: bool):
    source = """
      auto round_mantisa(at::Tensor x) -> void {{

        constexpr uint32_t o = 1;
        constexpr uint32_t n = {0};

        constexpr uint32_t trunc = {1};

        constexpr uint32_t _MANTISSA = {2};
        constexpr uint32_t _EXP_SIGN = {3};

        constexpr uint32_t mask = ((o << (n + _EXP_SIGN)) - o) << (_MANTISSA - n);
        constexpr uint32_t skip = trunc ? 0 : (o << (_MANTISSA - n - o));

        auto N = x.numel();
        auto * data = x.data_ptr<uint32_t>();

        for (int i = 0; i < N; i++) {{
            data[i] = (data[i] + skip) & mask;
        }}
      }}
    """.format(
        n,
        1 if trunc else 0,
        _MANTISSA,
        _EXP_SIGN,
    )

    cpp = load_inline(
        name="inline_extension",
        cpp_sources=source,
        functions="round_mantisa",
    )

    return cpp.round_mantisa  # type: ignore


def round_mantisa(x: torch.Tensor, n: int, trunc=False):
    """
    Take a tensor of float32 and round its mantissa to n bits.

    This runs on the CPU and will incur two copies if the input is on GPU.
    """
    assert x.dtype == torch.float32
    assert 0 <= n <= _MANTISSA

    if n == _MANTISSA:
        return x

    device = x.device

    x = x.cpu().clone(memory_format=torch.contiguous_format)

    assert x.is_contiguous()

    x = x.view(dtype=torch.uint32)
    _round_impl(n, trunc=trunc)(x)
    x = x.view(dtype=torch.float32)

    return x.to(device=device)


def lookup(
    x: torch.Tensor, fn, i_bits: Optional[int] = None, f_bits: int = 7, o_bits: int = 23
) -> torch.Tensor:
    """
    Evaluate fn on x emulating a vollo-esc lookup table.

    - Arguments:

        fn: is (Tensor -> Tensor), input is float64, output will be cast to float32.

        i_bits: number of bits for the integer part
        f_bits: number of bits for the fractional part
        o_bits: number of bits for the output mantissa (max 23)
    """

    kwargs = dict(
        dtype=x.dtype,
        device=x.device,
    )

    # Emulate n-bit fractional representation
    x = x.to(device="cpu", dtype=torch.float64)
    x = x * (2**f_bits)
    x = x.round()
    x = x / (2**f_bits)

    if i_bits is not None:
        assert torch.all(x.abs() < 2**i_bits)

    # Full precision fn on emulated fixed point
    x = fn(x)

    # Post-rounding (emulate bfN in look up table)
    x = x.to(torch.float32)
    x = round_mantisa(x, n=o_bits)
    x = x.to(**kwargs)

    return x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (lookup(x * 0.5, fn=F.tanh, f_bits=7, o_bits=23) + 1)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * sigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return lookup(x, fn=F.softplus, f_bits=7, o_bits=23)
