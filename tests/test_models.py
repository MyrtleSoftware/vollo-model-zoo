import pytest

from vollo_model_zoo.main import get_available_models, get_model_results
from vollo_model_zoo.vm import Result


@pytest.mark.parametrize("model_name", get_available_models())
def test_models(model_name: str):
    results = list(get_model_results(model_name))
    assert len(results) > 0, f"Model {model_name} returned no results"

    for result in results:
        assert isinstance(result, Result), (
            f"Model '{model_name}' produced a non-Result object: {type(result)}"
        )
