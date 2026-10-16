"""
The zoo's models and their sizes, as data.

Each model file lists the sizes its `main()` sweeps in a top-level `SIZES`, and
builds any one of them with `instance(**size, config=...)`. This module reads
`SIZES` from the file's source rather than importing it, so the sizes can be
listed without torch or the Vollo SDK installed:

    python -m vollo_model_zoo.catalog   # every model's sizes, as JSON

and a tool can then build and compile them one at a time with `build`.

That is also why this module imports only the standard library at module level
and, unlike the rest of the zoo, uses no `@beartype`.
"""

import ast
import importlib
import json
import os
import re
import sys
from typing import Any

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# The only names a `SIZES` expression can use.
_SIZES_BUILTINS = {"dict": dict, "list": list, "range": range, "tuple": tuple}

# The keys of a size that are requirements on the config, not arguments to
# `instance`.
_REQUIREMENTS = {
    # Compiler features the config's fabric must provide, e.g. "fp8".
    "_features",
    # The fewest cores the config may have.
    "_min_cores",
}


def models() -> list[str]:
    """
    Every model, by name: the `.py` files in `models/` not starting with `__`.
    """
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    return sorted(
        f.removesuffix(".py")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".py") and not f.startswith("__")
    )


def sizes(model: str) -> list[dict[str, Any]]:
    """
    A model's `SIZES`, evaluated from its file's source.
    """
    path = os.path.join(MODELS_DIR, f"{model}.py")
    with open(path) as f:
        tree = ast.parse(f.read(), path)

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SIZES"
        ):
            code = compile(ast.Expression(node.value), path, "eval")
            try:
                return eval(code, {"__builtins__": _SIZES_BUILTINS})
            except NameError as e:
                raise ValueError(
                    f"{path}: `SIZES` must be plain data, using no names but"
                    f" {sorted(_SIZES_BUILTINS)}: {e}"
                ) from e

    raise ValueError(f"{path} has no top-level `SIZES = ...`")


def split(size: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    A size's arguments to `instance`, and its requirements on the config.
    """
    unknown = {k for k in size if k.startswith("_")} - _REQUIREMENTS
    if unknown:
        raise ValueError(f"Unknown size requirements: {sorted(unknown)}")

    arguments = {k: v for k, v in size.items() if not k.startswith("_")}
    requirements = {k: v for k, v in size.items() if k.startswith("_")}
    return arguments, requirements


def runs_on(size: dict[str, Any], config: str) -> bool:
    from vollo_model_zoo.vm import CONFIGS, config_supports

    if config not in CONFIGS:
        raise ValueError(f"Unknown config: {config}, valid configs are: {[*CONFIGS]}")

    _, requirements = split(size)
    if not all(
        config_supports(config, feature)
        for feature in requirements.get("_features", [])
    ):
        return False
    if "_min_cores" in requirements:
        return CONFIGS[config].num_cores >= requirements["_min_cores"]
    return True


def size_name(size: dict[str, Any]) -> str:
    """
    A size's name, made from its arguments in the order they are written:
    `num_layers2-hidden_features512`. It uses only letters, digits, `_` and
    `-`, so it can be part of a file name.
    """
    arguments, _ = split(size)
    parts = []
    for key, value in arguments.items():
        value = str(value).replace(".", "p")
        parts.append(key + "x".join(re.findall(r"[A-Za-z0-9]+", value)))
    return "-".join(parts)


def named_sizes(model: str) -> dict[str, dict[str, Any]]:
    """
    A model's sizes by name, in `SIZES` order.
    """
    named = {}
    for size in sizes(model):
        name = size_name(size)
        if name in named:
            raise ValueError(f"{model}: two sizes are named {name}")
        named[name] = size
    return named


def build(model: str, name: str, config: str, *, seed: int = 0) -> dict[str, Any]:
    """
    The size `name` of `model`, built for `config` by the model's `instance`:
    the arguments, other than `config`, of `vollo_info`, or of
    `vollo_multi_model_info` for a multi-model file.

    Torch's RNG is seeded with `seed` first, so the weights and inputs don't
    depend on what was built before, and restored after, so the caller's RNG is
    left as it was. `main()` doesn't reseed between sizes, so its sweep
    compiles different weights.
    """
    import torch

    named = named_sizes(model)
    if name not in named:
        raise ValueError(f"{model} has no size {name}; its sizes are: {[*named]}")
    size = named[name]
    if not runs_on(size, config):
        raise ValueError(f"{model} size {name} can't run on {config}")

    module = importlib.import_module(f"vollo_model_zoo.models.{model}")

    arguments, _ = split(size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return module.instance(**arguments, config=config)


def listing() -> dict[str, list[dict[str, Any]]]:
    """
    Every model's sizes, in `SIZES` order, as JSON-ready data:
    `{model: [{"name", "arguments", "features", "min_cores"}, ...]}`.
    """
    entries = {}
    for model in models():
        entries[model] = []
        for name, size in named_sizes(model).items():
            arguments, requirements = split(size)
            entries[model].append(
                {
                    "name": name,
                    "arguments": arguments,
                    "features": requirements.get("_features", []),
                    "min_cores": requirements.get("_min_cores", 1),
                }
            )
    return entries


if __name__ == "__main__":
    json.dump(listing(), sys.stdout, indent=2)
    print()
