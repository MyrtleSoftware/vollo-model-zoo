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

Tools outside the zoo may use only these, so they change only for very good
reason:

- `python -m vollo_model_zoo.catalog`, and `listing()`, which it prints;
- `build(model, name, config, *, seed=0)`.

The rest of this module is the zoo's own. Which configs provide a size's
`_features` is `vm.runs_on`: it needs the Vollo SDK to know.
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
    # Whether the size is only swept on request (`zoo --experimental`).
    "_experimental",
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
    features = size.get("_features", [])
    if not (isinstance(features, list) and all(isinstance(f, str) for f in features)):
        raise ValueError(f"`_features` must be a list of strings, not {features!r}")
    if not isinstance(size.get("_experimental", False), bool):
        raise ValueError(
            f"`_experimental` must be a bool, not {size['_experimental']!r}"
        )

    arguments = {k: v for k, v in size.items() if not k.startswith("_")}
    requirements = {k: v for k, v in size.items() if k.startswith("_")}
    return arguments, requirements


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
    `vollo_multi_model_info` for a multi-model file. It doesn't check that
    `config` provides the size's features (see `vm.runs_on`); compiling one it
    doesn't fails.

    Torch's CPU RNG is seeded with `seed` first, so the weights and inputs don't
    depend on what was built before, and restored after, so the caller's RNG is
    left as it was. `main()` doesn't reseed between sizes, so its sweep
    compiles different weights.
    """
    import torch

    named = named_sizes(model)
    if name not in named:
        raise ValueError(f"{model} has no size {name}; its sizes are: {[*named]}")
    module = importlib.import_module(f"vollo_model_zoo.models.{model}")

    arguments, _ = split(named[name])
    with torch.random.fork_rng(devices=[]):
        # Not `torch.manual_seed`, which would also reseed every GPU, whose
        # RNGs `fork_rng(devices=[])` doesn't restore.
        torch.default_generator.manual_seed(seed)
        return module.instance(**arguments, config=config)


def listing() -> dict[str, list[dict[str, Any]]]:
    """
    Every model's sizes, in `SIZES` order, as JSON-ready data:
    `{model: [{"name", "arguments", "features", "experimental"}, ...]}`.
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
                    "experimental": requirements.get("_experimental", False),
                }
            )
    return entries


if __name__ == "__main__":
    json.dump(listing(), sys.stdout, indent=2)
    print()
