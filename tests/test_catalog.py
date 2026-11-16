import ast
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch
from beartype import beartype

from vollo_model_zoo import catalog, vm
from vollo_model_zoo.vm import CONFIGS

_MODELS = catalog.models()


@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_sizes_match_import(model_name: str):
    """
    Reading `SIZES` and `EXPERIMENTAL` from the source gives what importing the
    file does. This is what holds a file to one top-level assignment of each:
    the catalog reads only the first.
    """
    module = importlib.import_module(f"vollo_model_zoo.models.{model_name}")
    assert catalog.sizes(model_name) == module.SIZES
    assert catalog.experimental(model_name) == getattr(module, "EXPERIMENTAL", None)


def test_sweep(monkeypatch):
    """
    `sweep` yields a size's arguments without its `_` keys, in order, leaving
    out the sizes the config lacks the features for, and the experimental ones
    unless asked.
    """
    sizes = [dict(a=1), dict(a=2, _features=["fp8"]), dict(a=3, _experimental="x")]
    assert [*vm.sweep(sizes, "V80")] == [dict(a=1), dict(a=2)]
    assert [*vm.sweep(sizes, "IA-420f")] == [dict(a=1)]
    monkeypatch.setattr(vm, "sweep_experimental", True)
    assert [*vm.sweep(sizes, "V80")] == [dict(a=1), dict(a=2), dict(a=3)]


@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_imports_nothing_from_the_zoo(model_name: str):
    """
    A model file stands alone: it imports nothing from `vollo_model_zoo`, so
    `uv run <file>` runs it with only the PyPI packages its header lists.
    `nano-llm` builds on `mamba2`, so standalone it needs that file beside it.
    """
    with open(os.path.join(catalog.MODELS_DIR, f"{model_name}.py")) as f:
        source = f.read()
    assert "# /// script\n" in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." * node.level + (node.module or "")]
        else:
            continue
        for module in modules:
            if model_name == "nano-llm" and module in (".mamba2", "mamba2"):
                continue
            assert not module.startswith(("vollo_model_zoo", ".")), module


def _main_block(model_name: str) -> str:
    with open(os.path.join(catalog.MODELS_DIR, f"{model_name}.py")) as f:
        source = f.read()
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__main__'"
        ):
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{model_name} has no __main__ block")


@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_main_block_is_shared(model_name: str):
    """
    Every model file's `__main__` block is a verbatim copy of `mlp.py`'s, or
    `rnnt.py`'s for a multi-model file, so `test_runs_standalone` covers them all
    and a fix to one is made to every copy.
    """
    assert _main_block(model_name) in (_main_block("mlp"), _main_block("rnnt"))


@pytest.mark.parametrize("model_name", ["mlp", "rnnt"])
@beartype
def test_runs_standalone(model_name: str):
    """
    A model file runs as `python -c <source>`, as the Vollo sandbox runs it,
    with `vollo_model_zoo` unimportable: so no `__file__`, and nothing beside
    it to import. One single-model and one multi-model file; the others share
    their `__main__` (see `test_main_block_is_shared`).
    """
    with open(os.path.join(catalog.MODELS_DIR, f"{model_name}.py")) as f:
        source = f.read()
    script = "import sys; sys.modules['vollo_model_zoo'] = None\n" + source
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "us\n" in out.stdout


@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_build(model_name: str):
    """
    Every size has a name of its own, and builds, on the first config that can
    run it, into the arguments of `vollo_info` (or `vollo_multi_model_info`).
    """
    named = catalog.named_sizes(model_name)
    assert len(named) == len(catalog.sizes(model_name))

    for name, size in named.items():
        config = next(c for c in CONFIGS if vm.runs_on(size, c))
        instance = catalog.build(model_name, name, config)
        target = vm.vollo_multi_model_info if "entries" in instance else vm.vollo_info
        inspect.signature(target).bind(**instance, config=config)


def test_build_unknown_size():
    with pytest.raises(ValueError, match="has no size"):
        catalog.build("mlp", "nope", "V80")


def test_runs_on():
    def needing(model, requirement):
        return next(s for s in catalog.sizes(model) if requirement in s)

    fp8_rnnt = needing("rnnt", "_features")
    assert vm.runs_on(fp8_rnnt, "V80")
    assert not vm.runs_on(fp8_rnnt, "IA-420f")

    with pytest.raises(ValueError, match="Unknown config"):
        vm.runs_on(catalog.sizes("mlp")[0], "nope")


def test_boards():
    """
    A size compiled for several boards gets their cores, and the config it is
    a copy of is left as it was.
    """
    assert vm._config("V80", 4).num_cores == 4 * CONFIGS["V80"].num_cores
    assert vm._config("V80") is CONFIGS["V80"]


@pytest.mark.parametrize(
    "size",
    [
        dict(_features="fp8"),
        dict(_features=[8]),
        dict(_experimental=True),
        dict(_experimental=""),
        dict(_min_cores=24),
        dict(_x=1),
    ],
)
def test_split_rejects_bad_requirements(size):
    with pytest.raises(ValueError):
        catalog.split(size)


def test_build_is_seeded(monkeypatch):
    """
    A size's weights and inputs don't depend on what was built before it, and
    building leaves the caller's RNG as it was.
    """
    first, second = list(catalog.named_sizes("mlp"))[:2]
    a = catalog.build("mlp", second, "V80")
    catalog.build("mlp", first, "V80")

    state = torch.random.get_rng_state()
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: pytest.fail("`build` reseeded the GPUs"),
    )
    b = catalog.build("mlp", second, "V80")
    assert (torch.random.get_rng_state() == state).all()

    for p, q in zip(a["model"].parameters(), b["model"].parameters()):
        assert (p == q).all()
    assert (a["x"] == b["x"]).all()


def test_size_name():
    assert catalog.size_name(dict(act="relux")) == "actrelux"
    assert catalog.size_name(dict(channels=[128] * 3, scale=0.5)) == (
        "channels128x128x128-scale0p5"
    )


def test_listing_needs_only_the_standard_library():
    """
    The sizes can be listed, as JSON, without importing torch, the Vollo SDK
    or beartype.
    """
    script = (
        "import json, sys\n"
        "from vollo_model_zoo import catalog\n"
        "json.dump(catalog.listing(), sys.stdout)\n"
        "assert not {'torch', 'vollo_compiler', 'vollo_torch', 'beartype'}"
        " & sys.modules.keys()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    listing = json.loads(result.stdout)
    assert listing.keys() == set(_MODELS)
    assert listing["nano-llm"]["experimental"]
    assert not listing["mlp"]["experimental"]
    assert listing["nano-llm"]["sizes"][-1]["experimental"]
