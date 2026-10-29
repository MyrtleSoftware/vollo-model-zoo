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
    Reading `SIZES` from the source gives what importing the file does. This is
    what holds a file to one top-level `SIZES = ...`: the catalog reads only
    the first.
    """
    module = importlib.import_module(f"vollo_model_zoo.models.{model_name}")
    assert catalog.sizes(model_name) == module.SIZES


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("config", [*CONFIGS])
@pytest.mark.parametrize("model_name", _MODELS)
@beartype
def test_main_sweeps_sizes(
    model_name: str, config: str, experimental: bool, monkeypatch
):
    """
    `main()` builds exactly the sizes `config` can run, in `SIZES` order, and
    the experimental ones only when asked. Nothing is built or compiled:
    `instance` and the compile functions are replaced with recorders.
    """
    module = importlib.import_module(f"vollo_model_zoo.models.{model_name}")
    monkeypatch.setattr(vm, "sweep_experimental", experimental)
    calls = []
    monkeypatch.setattr(module, "instance", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(vm, "vollo_info", lambda **kw: None)
    monkeypatch.setattr(vm, "vollo_multi_model_info", lambda **kw: iter(()))

    list(module.main(config=config))

    assert calls == [
        {**catalog.split(size)[0], "config": config}
        for size in catalog.sizes(model_name)
        if vm.runs_on(size, config)
        and (experimental or not catalog.split(size)[1].get("_experimental"))
    ]


def test_copied_file_sweeps_its_own_sizes(tmp_path, monkeypatch):
    """
    A model file copied out of the zoo, renamed and with its `SIZES` edited,
    sweeps the sizes it lists.
    """
    source = os.path.join(catalog.MODELS_DIR, "mlp.py")
    with open(source) as f:
        text = f.read()
    copy = tmp_path / "my_mlp.py"
    copy.write_text(text + "\nSIZES = SIZES[:1]\n")
    spec = importlib.util.spec_from_file_location("my_mlp", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []
    monkeypatch.setattr(module, "instance", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(vm, "vollo_info", lambda **kw: None)
    list(module.main(config="V80"))

    assert calls == [{**catalog.split(catalog.sizes("mlp")[0])[0], "config": "V80"}]


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
        dict(_experimental=1),
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
    assert json.loads(result.stdout).keys() == set(_MODELS)
