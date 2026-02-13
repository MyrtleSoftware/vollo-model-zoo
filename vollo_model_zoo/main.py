import argparse
import importlib
import os
import sys


def get_available_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    models = []
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                models.append(filename[:-3])
    return sorted(models)


def main():
    available_models = get_available_models()

    parser = argparse.ArgumentParser(description="Run latency tests for Vollo models")
    parser.add_argument(
        "model",
        choices=available_models,
        help=f"Model to run. Available: {', '.join(available_models)}",
    )

    args = parser.parse_args()

    model_module_path = f"vollo_model_zoo.models.{args.model}"
    try:
        model_module = importlib.import_module(model_module_path)
        if hasattr(model_module, "main"):
            model_module.main()
        else:
            print(
                f"Error: Model module '{model_module_path}' does not have a main() function."
            )
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Could not import model '{args.model}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
