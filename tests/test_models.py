from itertools import pairwise
from typing import Optional

import pytest
from beartype import beartype
from vollo_compiler import AllocationError

from vollo_model_zoo.main import get_available_models, get_model_results
from vollo_model_zoo.vm import CONFIGS, Result


def is_sorted(xs, *, key):
    return all(a <= b for a, b in pairwise(map(key, xs)))


@pytest.mark.parametrize("model_name", get_available_models())
@pytest.mark.parametrize("config", [None, *CONFIGS.keys()])
@beartype
def test_models(model_name: str, config: Optional[str]):
    try:
        results = list(get_model_results(model_name, config=config))
    except AllocationError:
        # Skip test if not V80 or default config
        if config not in (None, "V80", "V80LL"):
            pytest.skip(f"Can't allocated on config {config}")
        raise

    assert len(results) > 0, f"Model {model_name} returned no results"

    param_sorted = is_sorted(results, key=lambda r: r.param_count)
    speed_sorted = is_sorted(results, key=lambda r: r.latency_spaced.microseconds)

    assert param_sorted or speed_sorted

    for result in results:
        assert isinstance(
            result, Result
        ), f"Model '{model_name}' produced a non-Result object: {type(result)}"
