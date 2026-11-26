from collections import defaultdict
from itertools import pairwise

import pytest
from beartype import beartype
from vollo_compiler import AllocationError, SaveError

from vollo_model_zoo import catalog, vm
from vollo_model_zoo.vm import CONFIGS, Ok, get_results


def is_sorted(xs, *, key):
    return all(a <= b for a, b in pairwise(map(key, xs)))


# Every model, including experimental, is tested on every config
_MODELS = catalog.models()


@pytest.mark.parametrize("config", [*CONFIGS])
@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_models(model_name: str, config: str, monkeypatch):
    # Experimental sizes too, on every config that can run them.
    monkeypatch.setattr(vm, "sweep_experimental", True)
    results: list[Ok] = []

    for r in get_results(model_name, config=config):
        match r:
            case Ok():
                results.append(r)
            case AllocationError():
                if config in ("V80", "V80LL"):
                    raise r
            case SaveError():
                if config in ("V80", "V80LL"):
                    raise r
            case _:
                raise ValueError(f"Unexpected result type: {type(r)}")

    assert len(results) > 0, f"Model {model_name} returned no results"

    # Each entry of a multi-model file (its `prog`) is a sweep of its own.
    sweeps = defaultdict(list)
    for r in results:
        sweeps[(r.meta or {}).get("prog")].append(r)
    for sweep in sweeps.values():
        param_sorted = is_sorted(sweep, key=lambda r: r.param_count)
        speed_sorted = is_sorted(sweep, key=lambda r: r.latency_spaced.microseconds)
        assert param_sorted or speed_sorted

    # There should be at least one result close to 1-mill parameters, aka the "baseline"
    assert any(0.95e6 < r.param_count < 1.05e6 for r in results)
