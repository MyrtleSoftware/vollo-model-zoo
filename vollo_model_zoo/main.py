import argparse
import importlib
import os

from beartype import beartype


@beartype
def get_available_models() -> list[str]:
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    @beartype
    def is_valid_model_file(filename: str) -> bool:
        return filename.endswith(".py") and not filename.startswith("__")

    return sorted(f[:-3] for f in os.listdir(models_dir) if is_valid_model_file(f))


@beartype
def main() -> None:
    available_models = get_available_models()

    parser = argparse.ArgumentParser(description="Run latency tests for Vollo models")

    parser.add_argument("model", choices=available_models, help="Model to run")

    args = parser.parse_args()

    model_module_path = f"vollo_model_zoo.models.{args.model}"

    model_module = importlib.import_module(model_module_path)

    if hasattr(model_module, "main"):
        model_module.main()
    else:
        raise ImportError(
            f"Model module '{model_module_path}' does not have a main() function."
        )


if __name__ == "__main__":
    main()
