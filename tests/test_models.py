import importlib

import pytest

from vollo_model_zoo.main import get_available_models
from vollo_model_zoo.vm import Result


@pytest.mark.parametrize("model_name", get_available_models())
def test_models(model_name: str):
    model_module_path = f"vollo_model_zoo.models.{model_name}"
    model_module = importlib.import_module(model_module_path)

    assert hasattr(model_module, "main"), f"Model {model_name} has no main() function"

    results = list(model_module.main())
    assert len(results) > 0, f"Model {model_name} returned no results"

    for result in results:
        assert isinstance(result, Result), (
            f"Model '{model_name}' produced a non-Result object: {type(result)}"
        )
