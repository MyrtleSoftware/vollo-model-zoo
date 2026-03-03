from itertools import pairwise
from typing import Optional

import pytest
from beartype import beartype
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo.main import get_available_models, get_model_results
from vollo_model_zoo.vm import CONFIGS, Ok


def is_sorted(xs, *, key):
    return all(a <= b for a, b in pairwise(map(key, xs)))


def idfn(config):
    if config is None:
        return "default"
    return config


@pytest.mark.parametrize("config", [None, *CONFIGS.keys()], ids=idfn)
@pytest.mark.parametrize("model_name", get_available_models())
@beartype
def test_models(model_name: str, config: Optional[str]):
    #
    results: list[Ok] = []

    for r in get_model_results(model_name, config=config):
        if not isinstance(r, Ok):
            if config not in (None, "V80", "V80LL"):
                pytest.skip(f"Can't allocated on config {config}")
            raise r

        results.append(r)

    assert len(results) > 0, f"Model {model_name} returned no results"

    param_sorted = is_sorted(results, key=lambda r: r.param_count)
    speed_sorted = is_sorted(results, key=lambda r: r.latency_spaced.microseconds)

    assert param_sorted or speed_sorted

    # There should be at least one result close to 1-mill parameters, aka the "baseline"
    assert any(map(lambda r: 0.95e6 < r.param_count < 1.05e6, results))
