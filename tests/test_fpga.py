"""
Tests of the hardware path (`vollo_model_zoo/fpga.py`) without a card: config
detection, compiling to programs, and assembling the results JSON.

The `vollo-example --json` output the collector reads is faked here in the shape
`fpga.py` assumes: `metrics.latency_us.{mean,median,p99}`.
"""

import inspect
import json
from pathlib import Path

import pytest
import vollo_compiler as vc

from vollo_model_zoo import catalog, fpga, vm
from vollo_model_zoo.parse import parse_records_from_json
from vollo_model_zoo.version import benchmark_version

VERSION = "0.0.0+zoo.0.0.0"
# The config card results are under.
V80_FPGA = "V80" + fpga.FPGA_SUFFIX


def _save_config(config: vc.Config, path):
    config.save(str(path))
    return path


@pytest.fixture
def hw_config(tmp_path):
    return _save_config(vm.CONFIGS["V80"], tmp_path / "hw_config.json")


def _slp_sizes() -> list[str]:
    return [size["name"] for size in catalog.listing()["slp"]["sizes"]]


def _skip_all_but(model: str) -> list[str]:
    return [arg for m in catalog.models() if m != model for arg in ("--skip", m)]


def _compile_argv(hw_config, out, config: str = "V80") -> list[str]:
    return ["--config", config, "--hw-config", str(hw_config), "--out", str(out)]


def _listed(out) -> list[list[str]]:
    return [line.split() for line in (out / "sizes.txt").read_text().splitlines()]


# Config detection


@pytest.mark.parametrize("name", list(vm.CONFIGS))
def test_detect_config_of_every_zoo_config(name, tmp_path):
    """A card reporting exactly a zoo config is detected as it, twins included."""
    path = _save_config(vm.CONFIGS[name], tmp_path / "hw_config.json")
    assert fpga.detect_config(vc.Config.load(str(path))) == name


@pytest.mark.parametrize("name", ["V80", "V80LL"])
def test_detect_config_tells_v80_from_v80ll_by_clock(name, tmp_path):
    """V80 and V80LL differ only in name and clock; a renamed card is told by clock."""
    path = _save_config(vm.CONFIGS[name], tmp_path / "hw_config.json")
    loaded = vc.Config.load(str(path))
    loaded.name = "renamed-by-some-release"
    assert fpga.detect_config(loaded) == name


def test_detect_config_ignores_a_name_and_clock_it_does_not_know(hw_config):
    """The hardware matching alone is enough when only one config is that hardware."""
    loaded = vc.Config.load(str(_save_config(vm.CONFIGS["IA-840f"], hw_config)))
    loaded.name = "renamed-by-some-release"
    loaded.clock_mhz = 123.0
    assert fpga.detect_config(loaded) == "IA-840f"


def test_detect_config_is_an_error_when_twins_cannot_be_told_apart(tmp_path):
    """IA-420f and NT400D11 are one hardware at one clock: nothing to choose on."""
    if not {"IA-420f", "NT400D11"} <= vm.CONFIGS.keys():
        pytest.skip("this SDK lacks one of the twin configs")
    path = _save_config(vm.CONFIGS["IA-420f"], tmp_path / "hw_config.json")
    loaded = vc.Config.load(str(path))
    loaded.name = "renamed-by-some-release"
    with pytest.raises(ValueError, match="IA-420f.*NT400D11"):
        fpga.detect_config(loaded)


def test_config_main_prints_the_config_or_fails(hw_config, tmp_path, capsys):
    assert fpga.config_main(["--hw-config", str(hw_config)]) == 0
    assert capsys.readouterr().out.strip() == "V80"

    loaded = vc.Config.load(str(hw_config))
    loaded.num_cores += 1
    odd = _save_config(loaded, tmp_path / "odd.json")
    assert fpga.config_main(["--hw-config", str(odd)]) == 1
    assert "no zoo config" in capsys.readouterr().err


# Compiling


def test_compile_only_what_is_asked_for(tmp_path, hw_config):
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    sizes = _listed(out)
    assert sizes
    assert any((out / f"{m}-{s}.vollo").exists() for m, s in sizes)
    # Skipped models are left out entirely, not listed as skipped.
    assert {model for model, _ in sizes} == {"slp"}
    for model, size in sizes:
        name = f"{model}-{size}"
        compiled = (out / f"{name}.vollo").exists()
        error = out / f"{name}.error.json"
        # Exactly one outcome per size, and the zoo's own slp never fails to
        # compile, though a size may not fit.
        assert compiled != error.exists(), name
        if error.exists():
            assert json.loads(error.read_text()) in (
                {"AllocationError": 0},
                {"SaveError": 1},
            )
        if compiled:
            assert (out / f"{name}.num-models").read_text().strip() == "1"
            (info,) = json.loads((out / f"{name}.meta.json").read_text())
            assert info["param_count"] > 0
            assert info["cycle_count"] > 0


def test_compile_skips_one_size(tmp_path, hw_config):
    first, *_ = _slp_sizes()
    out = tmp_path / "programs"
    argv = _compile_argv(hw_config, out) + _skip_all_but("slp")
    assert fpga.compile_main(argv + ["--skip", f"slp-{first}"]) == 0
    assert {size for _, size in _listed(out)} == set(_slp_sizes()) - {first}


def test_compile_rejects_a_skip_naming_nothing(tmp_path, hw_config, capsys):
    out = tmp_path / "programs"
    argv = _compile_argv(hw_config, out) + _skip_all_but("slp")
    with pytest.raises(SystemExit) as exit:
        fpga.compile_main(argv + ["--skip", "slp-no-such-size"])
    assert exit.value.code != 0
    assert "slp-no-such-size" in capsys.readouterr().err
    # Nothing was compiled.
    assert not (out / "sizes.txt").exists()


def test_compile_sweeps_experimental_models_but_not_sizes(tmp_path, hw_config):
    """As the VM benchmarks: an experimental model is compiled, and every
    model's experimental sizes are left out."""
    listing = catalog.listing()
    experimental_sizes = {
        (model, size["name"])
        for model, entry in listing.items()
        for size in entry["sizes"]
        if size["experimental"] is not None
    }
    experimental = [m for m, entry in listing.items() if entry["experimental"]]
    # Compile only the models concerned, so the test stays cheap.
    concerned = set(experimental) | {model for model, _ in experimental_sizes}
    if not concerned:
        pytest.skip("no model or size is experimental at the moment")
    skips = [
        arg
        for model in catalog.models()
        if model not in concerned
        for arg in ("--skip", model)
    ]
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + skips) == 0
    listed = {tuple(line) for line in _listed(out)}
    assert not experimental_sizes & listed
    for model in experimental:
        named = catalog.named_sizes(model)
        expected = {
            (model, size["name"])
            for size in listing[model]["sizes"]
            if size["experimental"] is None and vm.runs_on(named[size["name"]], "V80")
        }
        assert expected and expected <= listed


def test_compile_checks_features_on_the_cards_hw_config(
    tmp_path, hw_config, monkeypatch
):
    """A size needing a feature the card lacks is left out, even if the zoo's
    config of the same name has it."""
    asked = []

    def runs_on(size, config):
        asked.append(config)
        return False

    monkeypatch.setattr(vm, "runs_on", runs_on)
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0
    assert asked and all(isinstance(config, vc.Config) for config in asked)
    assert asked[0] == vc.Config.load(str(hw_config))
    assert _listed(out) == []


@pytest.mark.parametrize(
    "error, message, entry",
    [
        # Doesn't fit, as `vm.to_dict` writes it.
        (vc.AllocationError, "too big", {"AllocationError": 0}),
        (vc.SaveError, "too big", {"SaveError": 1}),
        # As the VM records it.
        (
            ValueError,
            "unsupported function 'fft_fft': found 1",
            {"ValueError": "unsupported function 'fft_fft': found 1"},
        ),
        # Any other failure, with the whole message, not only its last line.
        (
            RuntimeError,
            "boom\non two lines",
            {"CompileError": "RuntimeError: boom\non two lines"},
        ),
        # An `assert` failing inside the SDK is that size's CompileError, not
        # the end of the run.
        (
            AssertionError,
            "an assert inside vollo_torch",
            {"CompileError": "AssertionError: an assert inside vollo_torch"},
        ),
    ],
)
def test_compile_records_a_size_that_fails(
    error, message, entry, tmp_path, hw_config, monkeypatch
):
    def fails(zoo, config):
        raise error(message)

    monkeypatch.setattr(vm, "compile_instance", fails)
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0
    for model, size in _listed(out):
        assert not (out / f"{model}-{size}.vollo").exists()

    runs = tmp_path / "runs"
    runs.mkdir()
    entries = fpga.collect(out, runs, "V80", VERSION)[VERSION]["slp"][V80_FPGA]
    assert entries == [entry] * len(_slp_sizes())


class _PartSavedProgram:
    """A compiled program whose `save` writes part of the `.vollo`, then fails."""

    def num_models(self):
        return 1

    def save(self, path):
        Path(path).write_bytes(b"part of a program")
        raise RuntimeError("bogus save")


def test_compile_leaves_no_program_for_a_size_that_fails_to_save(
    tmp_path, hw_config, monkeypatch
):
    """scripts/fpga-benchmark.sh runs every `.vollo` it finds, so a size
    recorded as an error must not leave one (or its other files) behind."""
    monkeypatch.setattr(vm, "compile_instance", lambda zoo, config: _PartSavedProgram())
    monkeypatch.setattr(vm, "program_results", lambda zoo, program, config: [])
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    for model, size in _listed(out):
        name = f"{model}-{size}"
        assert json.loads((out / f"{name}.error.json").read_text()) == {
            "CompileError": "RuntimeError: bogus save"
        }
        for suffix in (".vollo", ".num-models", ".meta.json"):
            assert not (out / f"{name}{suffix}").exists(), suffix


@pytest.mark.parametrize(
    "error",
    [
        # `compile_instance` rejecting a size: the zoo's limitation.
        vm.ZooArgumentError("Can't compile for bogus boards"),
        # The machine running out of memory, not anything the SDK did.
        MemoryError("bogus"),
    ],
)
def test_compile_does_not_hide_an_error_as_a_compile_error(
    error, tmp_path, hw_config, monkeypatch
):
    def raises(zoo, config):
        raise error

    monkeypatch.setattr(vm, "compile_instance", raises)
    out = tmp_path / "programs"
    with pytest.raises(type(error), match="bogus"):
        fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp"))


# The types `_compile_size` would otherwise record as a size's compile result.
@pytest.mark.parametrize("error", [AssertionError("bogus"), ValueError("bogus")])
def test_compile_does_not_hide_a_bug_building_a_size(
    error, tmp_path, hw_config, monkeypatch
):
    """Building a size is the zoo's own code, so anything it raises is a zoo
    bug, as on the VM path, not a size's compile result."""

    def build(model, size, config):
        raise error

    monkeypatch.setattr(catalog, "build", build)
    out = tmp_path / "programs"
    with pytest.raises(type(error), match="bogus"):
        fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp"))


def test_compile_multi_model_program(tmp_path, hw_config):
    """A multi-model size is one program with an info per model."""
    first, *rest = [s["name"] for s in catalog.listing()["rnnt"]["sizes"]]
    skips = [arg for size in rest for arg in ("--skip", f"rnnt-{size}")]
    out = tmp_path / "programs"
    argv = _compile_argv(hw_config, out) + _skip_all_but("rnnt") + skips
    assert fpga.compile_main(argv) == 0

    assert _listed(out) == [["rnnt", first]]
    name = f"rnnt-{first}"
    assert (out / f"{name}.vollo").exists()
    infos = json.loads((out / f"{name}.meta.json").read_text())
    assert len(infos) > 1
    assert int((out / f"{name}.num-models").read_text()) == len(infos)
    assert all(info["cycle_count"] > 0 for info in infos)
    assert len({info["meta"]["prog"] for info in infos}) == len(infos)


def test_program_results_match_vollo_info():
    """The saved program is the one the zoo measures: what it records about a
    size is what the VM benchmarks do."""
    name = _slp_sizes()[0]
    zoo = catalog.build("slp", name, "V80")
    program = vm.compile_instance(zoo, vm.CONFIGS["V80"])
    (result,) = vm.program_results(zoo, program, "V80")
    assert result == vm.vollo_info(**catalog.build("slp", name, "V80"), config="V80")


def _named_parameters(fn) -> set[str]:
    return {
        parameter.name
        for parameter in inspect.signature(fn).parameters.values()
        if parameter.kind is not parameter.VAR_KEYWORD
    }


def test_compile_instance_compiles_every_argument_vollo_info_takes():
    """A size's arguments are `vollo_info`'s (`test_catalog.py`), so the
    functions `compile_instance` hands them to must name every one it doesn't
    handle itself: one `vollo_info` gained would otherwise go on through
    `_vollo_compile`'s `**kwargs` to `to_nnir`, and fail there as if the SDK
    had."""
    handled = {"meta", "boards"}
    assert _named_parameters(vm.vollo_info) - handled <= _named_parameters(
        vm._vollo_compile
    )
    assert _named_parameters(vm.vollo_multi_model_info) <= _named_parameters(
        vm._compile_entries
    )


def test_compile_instance_rejects_several_boards():
    zoo = catalog.build("slp", _slp_sizes()[0], "V80")
    with pytest.raises(vm.ZooArgumentError, match="4 boards"):
        vm.compile_instance({**zoo, "boards": 4}, vm.CONFIGS["V80"])
    # One board is the config itself.
    vm.compile_instance({**zoo, "boards": 1}, vm.CONFIGS["V80"])


# Collecting


def _write_run(runs, name, index, mean, median, p99, contiguous_mean=None):
    """What `vollo-example --json` prints, as `fpga.py` reads it."""
    for kind, kind_mean in (("contiguous", contiguous_mean), ("spaced", None)):
        latency = dict(mean=mean if kind_mean is None else kind_mean)
        metrics = dict(latency_us=dict(latency, median=median, p99=p99))
        (runs / f"{name}.{index}.{kind}.json").write_text(
            json.dumps(dict(metrics=metrics))
        )


@pytest.fixture
def fake_results(tmp_path):
    """One of every outcome a size can have, as the script leaves them."""
    programs, runs = tmp_path / "programs", tmp_path / "runs"
    programs.mkdir()
    runs.mkdir()
    info = dict(param_count=10, cycle_count=100, meta={"width": 4})
    sizes = [
        ("m", "ok"),
        ("m", "norun"),
        ("m", "nofit"),
        ("m", "nosave"),
        ("m", "broken"),
        ("m", "contiguousfail"),
        ("m", "spacedfail"),
        ("multi", "two"),
    ]
    (programs / "sizes.txt").write_text("".join(f"{m} {s}\n" for m, s in sizes))
    (programs / "m-ok.meta.json").write_text(json.dumps([info]))
    (programs / "m-norun.meta.json").write_text(json.dumps([info]))
    (programs / "m-nofit.error.json").write_text('{"AllocationError": 0}')
    (programs / "m-nosave.error.json").write_text('{"SaveError": 1}')
    (programs / "m-broken.error.json").write_text('{"ValueError": "unsupported op"}')
    (programs / "multi-two.meta.json").write_text(
        json.dumps([dict(info, meta={"prog": "a"}), dict(info, meta={"prog": "b"})])
    )

    _write_run(runs, "m-ok", 0, mean=1.5, median=1.4, p99=3.0, contiguous_mean=2.0)
    # One run of the two failed, as the script leaves it: the failed run's JSON
    # may be there too, but its run-error wins.
    for name, failed in (
        ("m-contiguousfail", "contiguous"),
        ("m-spacedfail", "spaced"),
    ):
        (programs / f"{name}.meta.json").write_text(json.dumps([info]))
        _write_run(runs, name, 0, mean=1.5, median=1.4, p99=3.0)
        (runs / f"{name}.0.{failed}.run-error").write_text(
            "vollo-example exited with 1\n"
        )
    (runs / "multi-two.0.contiguous.run-error").write_text("timed out after 600s\n")
    (runs / "multi-two.1.skipped").write_text("an earlier run timed out\n")
    return programs, runs


def test_collect_every_outcome(fake_results):
    programs, runs = fake_results
    data = fpga.collect(programs, runs, "V80", VERSION)
    assert list(data) == [VERSION]
    # Under their own config, so card results never pass for the VM's.
    assert list(data[VERSION]["m"]) == [V80_FPGA]
    (
        ok,
        norun,
        nofit,
        nosave,
        broken,
        contiguousfail,
        spacedfail,
    ) = data[VERSION]["m"][V80_FPGA]

    assert ok["Ok"]["config"] == V80_FPGA
    assert ok["Ok"]["param_count"] == 10
    assert ok["Ok"]["cycle_count"] == 100
    assert ok["Ok"]["meta"] == {"width": 4}
    # Each latency is the mean of its own run.
    assert ok["Ok"]["latency_spaced"] == {"microseconds": 1.5}
    assert ok["Ok"]["latency_contiguous"] == {"microseconds": 2.0}
    hardware = ok["Ok"]["hardware"]
    assert hardware["name"] == "m-ok"
    assert hardware["model_index"] == 0
    assert "hw_config" not in hardware
    assert hardware["contiguous"] == dict(mean=2.0, median=1.4, p99=3.0)
    assert hardware["spaced"] == dict(mean=1.5, median=1.4, p99=3.0)

    assert norun == {"RunError": "no run result"}
    assert nofit == {"AllocationError": 0}
    assert nosave == {"SaveError": 1}
    assert broken == {"ValueError": "unsupported op"}

    # One run failed: still Ok, with that run's latency null and its error in
    # `hardware`, and the other run's numbers kept.
    stats = dict(mean=1.5, median=1.4, p99=3.0)
    error = {"RunError": "vollo-example exited with 1"}
    assert contiguousfail["Ok"]["latency_contiguous"] is None
    assert contiguousfail["Ok"]["latency_spaced"] == {"microseconds": 1.5}
    assert contiguousfail["Ok"]["hardware"]["contiguous"] == error
    assert contiguousfail["Ok"]["hardware"]["spaced"] == stats
    assert spacedfail["Ok"]["latency_spaced"] is None
    assert spacedfail["Ok"]["latency_contiguous"] == {"microseconds": 1.5}
    assert spacedfail["Ok"]["hardware"]["contiguous"] == stats
    assert spacedfail["Ok"]["hardware"]["spaced"] == error

    # One entry per model of a multi-model program, each with its own outcome.
    assert data[VERSION]["multi"][V80_FPGA] == [
        {"RunError": "timed out after 600s"},
        {"Skipped": "an earlier run timed out"},
    ]


def test_collect_records_an_unexpected_run_output_as_its_run_error(tmp_path):
    """A run whose output is a JSON object without a number `collect` reads is
    that run's error, rather than failing the collection of every run."""
    programs, runs = tmp_path / "programs", tmp_path / "runs"
    programs.mkdir()
    runs.mkdir()
    info = dict(param_count=10, cycle_count=100, meta={"width": 4})
    sizes = ["nop99", "nullmean", "bothbad", "ok"]
    (programs / "sizes.txt").write_text("".join(f"m {s}\n" for s in sizes))
    for size in sizes:
        (programs / f"m-{size}.meta.json").write_text(json.dumps([info]))
        _write_run(runs, f"m-{size}", 0, mean=1.5, median=1.4, p99=3.0)

    # Contiguous lacks its p99; spaced is fine.
    run = runs / "m-nop99.0.contiguous.json"
    output = json.loads(run.read_text())
    del output["metrics"]["latency_us"]["p99"]
    run.write_text(json.dumps(output))
    # Spaced has its mean, but not as a number; contiguous is fine.
    run = runs / "m-nullmean.0.spaced.json"
    output = json.loads(run.read_text())
    output["metrics"]["latency_us"]["mean"] = None
    run.write_text(json.dumps(output))
    # Both are JSON, but neither is a run's output.
    (runs / "m-bothbad.0.contiguous.json").write_text("{}")
    (runs / "m-bothbad.0.spaced.json").write_text("[]")

    data = fpga.collect(programs, runs, "V80", VERSION)
    nop99, nullmean, bothbad, ok = data[VERSION]["m"][V80_FPGA]

    stats = dict(mean=1.5, median=1.4, p99=3.0)
    assert nop99["Ok"]["latency_contiguous"] is None
    assert nop99["Ok"]["latency_spaced"] == {"microseconds": 1.5}
    assert nop99["Ok"]["hardware"]["contiguous"] == {
        "RunError": "vollo-example output lacks metrics.latency_us.p99"
    }
    assert nop99["Ok"]["hardware"]["spaced"] == stats

    assert nullmean["Ok"]["latency_spaced"] is None
    assert nullmean["Ok"]["latency_contiguous"] == {"microseconds": 1.5}
    assert nullmean["Ok"]["hardware"]["contiguous"] == stats
    assert nullmean["Ok"]["hardware"]["spaced"] == {
        "RunError": "vollo-example output lacks metrics.latency_us.mean"
    }

    # Both failed: the entry is the contiguous run's error, as for any run error.
    assert bothbad == {
        "RunError": "vollo-example output lacks metrics.latency_us.mean, "
        "metrics.latency_us.p99"
    }

    # The sizes around them are unaffected.
    assert ok["Ok"]["hardware"]["contiguous"] == stats
    assert ok["Ok"]["hardware"]["spaced"] == stats


def test_collect_records_a_number_out_of_range_as_its_run_error(tmp_path):
    """A zero latency would report nonsense: it is that run's error, and the
    other sizes are still collected."""
    programs, runs = tmp_path / "programs", tmp_path / "runs"
    programs.mkdir()
    runs.mkdir()
    info = dict(param_count=10, cycle_count=100, meta={"width": 4})
    (programs / "sizes.txt").write_text("m zero\nm ok\n")
    for size in ("zero", "ok"):
        (programs / f"m-{size}.meta.json").write_text(json.dumps([info]))
    _write_run(runs, "m-zero", 0, mean=0, median=0, p99=3.0)
    _write_run(runs, "m-ok", 0, mean=1.5, median=1.4, p99=3.0)

    zero, ok = fpga.collect(programs, runs, "V80", VERSION)[VERSION]["m"][V80_FPGA]
    assert zero == {"RunError": "vollo-example output has metrics.latency_us.mean = 0"}
    assert ok["Ok"]["latency_contiguous"] == {"microseconds": 1.5}


def test_collect_writes_whole_number_latencies_as_floats(tmp_path):
    """A C program may print 3.0 as `3`; `parse` reads latencies as floats, so
    an int would cost `plot` and `report` the whole file."""
    programs, runs = tmp_path / "programs", tmp_path / "runs"
    programs.mkdir()
    runs.mkdir()
    info = dict(param_count=10, cycle_count=100, meta={"width": 4})
    (programs / "sizes.txt").write_text("m ints\n")
    (programs / "m-ints.meta.json").write_text(json.dumps([info]))
    _write_run(runs, "m-ints", 0, mean=3, median=2, p99=4)

    data = fpga.collect(programs, runs, "V80", VERSION)
    (ints,) = data[VERSION]["m"][V80_FPGA]
    assert ints["Ok"]["latency_spaced"] == {"microseconds": 3.0}
    assert type(ints["Ok"]["latency_spaced"]["microseconds"]) is float
    for kind in ("contiguous", "spaced"):
        assert all(type(v) is float for v in ints["Ok"]["hardware"][kind].values()), (
            kind
        )

    out = tmp_path / "results.json"
    out.write_text(json.dumps(data))
    (record,) = parse_records_from_json(out)
    assert record["latency_spaced"] == 3.0


def test_collect_main_writes_the_results_json(fake_results, tmp_path):
    programs, runs = fake_results
    out = tmp_path / "results.json"
    argv = [
        *("--config", "V80"),
        *("--programs", str(programs)),
        *("--runs", str(runs)),
        *("--out", str(out)),
    ]
    assert fpga.collect_main(argv) == 0
    # The shape of benchmarks/*.json: {version: {model: {config: [entry, ...]}}}.
    data = json.loads(out.read_text())
    assert list(data) == [benchmark_version()]
    assert set(data[benchmark_version()]) == {"m", "multi"}


def test_parse_reads_card_results_with_a_null_latency(fake_results, tmp_path):
    """A run that failed leaves a null latency, which must not cost `plot` and
    `report` the rest of the file."""
    programs, runs = fake_results
    out = tmp_path / "results.json"
    out.write_text(json.dumps(fpga.collect(programs, runs, "V80", VERSION)))
    records = parse_records_from_json(out)
    # Every Ok entry: ok, contiguousfail, spacedfail.
    assert len(records) == 3
    assert {record["config"] for record in records} == {V80_FPGA}
    assert sum(record["latency_spaced"] is None for record in records) == 1
    assert sum(record["latency_contiguous"] is None for record in records) == 1


def test_compile_then_collect_without_a_card(tmp_path, hw_config):
    """The pieces fit: every compiled size becomes an entry a run error can fill."""
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    runs = tmp_path / "runs"
    runs.mkdir()
    for model, size in _listed(out):
        name = f"{model}-{size}"
        if (out / f"{name}.vollo").exists():
            (runs / f"{name}.0.contiguous.run-error").write_text("no card")

    data = fpga.collect(out, runs, "V80", VERSION)
    entries = data[VERSION]["slp"][V80_FPGA]
    assert len(entries) == len(_slp_sizes())
    assert {"RunError": "no card"} in entries
    json.dumps(data)
