from itertools import pairwise
from typing import Optional

import pytest
from beartype import beartype
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo import catalog, vm
from vollo_model_zoo.vm import CONFIGS, Ok, get_results


def is_sorted(xs, *, key):
    return all(a <= b for a, b in pairwise(map(key, xs)))


def idfn(config):
    if config is None:
        return "default"
    return config


# Every model, including experimental, is tested on every config
_MODELS = catalog.models()


@pytest.mark.parametrize("config", [None, *CONFIGS], ids=idfn)
@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_models(model_name: str, config: Optional[str], monkeypatch):
    # The default config also sweeps the experimental sizes, so they are
    # compiled somewhere.
    monkeypatch.setattr(vm, "sweep_experimental", config is None)
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
