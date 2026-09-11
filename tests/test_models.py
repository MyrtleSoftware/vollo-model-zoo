from itertools import pairwise
from typing import Optional

import pytest
from beartype import beartype
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo.vm import (
    CONFIGS,
    EXPERIMENTAL_CONFIGS,
    Ok,
    get_models,
    get_results,
)


def is_sorted(xs, *, key):
    return all(a <= b for a, b in pairwise(map(key, xs)))


def idfn(config):
    if config is None:
        return "default"
    return config


# Experimental configs are restricted to one model each, so they aren't testable
# across every model — and the model an experimental config exists for is left
# out too, since the config it is written for is the one CI can't run. Check
# that pairing by hand: `zoo <model> --config <config> --experimental`.
_CONFIGS = [name for name in CONFIGS if name not in EXPERIMENTAL_CONFIGS]
_MODELS = [name for name in get_models() if name not in EXPERIMENTAL_CONFIGS.values()]


@pytest.mark.parametrize("config", [None, *_CONFIGS], ids=idfn)
@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_models(model_name: str, config: Optional[str]):
    #
    results: list[Ok] = []

    for r in get_results(model_name, config=config):
        match r:
            case Ok():
                results.append(r)
            case AllocationError():
                if config in (None, "V80", "V80LL"):
                    raise r
            case SaveError():
                if config in (None, "V80", "V80LL"):
                    raise r
            case _:
                raise ValueError(f"Unexpected result type: {type(r)}")

    assert len(results) > 0, f"Model {model_name} returned no results"

    param_sorted = is_sorted(results, key=lambda r: r.param_count)
    speed_sorted = is_sorted(results, key=lambda r: r.latency_spaced.microseconds)

    assert param_sorted or speed_sorted

    # There should be at least one result close to 1-mill parameters, aka the "baseline"
    assert any(map(lambda r: 0.95e6 < r.param_count < 1.05e6, results))
