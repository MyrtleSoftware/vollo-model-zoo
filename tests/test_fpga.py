"""
Tests of the hardware path (`vollo_model_zoo/fpga.py`) without a card: config
detection, compiling to programs, and assembling the results JSON.

The `vollo-example --json` output the collector reads is faked here in the shape
`fpga.py` assumes: `metrics.latency_us.{mean,median,p99}` and
`metrics.throughput` (inferences per second).
"""

import io
import json
import sys

import pytest
import vollo_compiler as vc

from vollo_model_zoo import catalog, fpga, vm
from vollo_model_zoo.version import benchmark_version

VERSION = "0.0.0+zoo.0.0.0"


def _save_config(config: vc.Config, path):
    if not hasattr(config, "save"):
        pytest.skip("this SDK can't save a Config")
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


def test_detect_config_is_an_error_for_unknown_hardware(hw_config):
    loaded = vc.Config.load(str(hw_config))
    loaded.num_cores += 1
    with pytest.raises(ValueError, match="no zoo config"):
        fpga.detect_config(loaded)


def test_config_main_prints_the_config_or_fails(hw_config, tmp_path, capsys):
    assert fpga.config_main(["--hw-config", str(hw_config)]) == 0
    assert capsys.readouterr().out.strip() == "V80"

    loaded = vc.Config.load(str(hw_config))
    loaded.num_cores += 1
    odd = _save_config(loaded, tmp_path / "odd.json")
    assert fpga.config_main(["--hw-config", str(odd)]) == 1
    assert "no zoo config" in capsys.readouterr().err


# The small commands of scripts/fpga-benchmark.sh


def test_hw_config_main_extracts_the_first_card(tmp_path, monkeypatch):
    cards = [{"hw_config": {"name": "first"}}, {"hw_config": {"name": "second"}}]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(cards)))
    out = tmp_path / "hw.json"
    assert fpga.hw_config_main(["--out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"name": "first"}


@pytest.mark.parametrize("stdin", ["[]", '[{"no_hw_config": 1}]'])
def test_hw_config_main_fails_without_a_hw_config(stdin, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    out = tmp_path / "hw.json"
    assert fpga.hw_config_main(["--out", str(out)]) == 1
    assert not out.exists()


def test_check_json_main(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text('{"metrics": {}}')
    bad.write_text('{"metrics": ')
    assert fpga.check_json_main([str(good)]) == 0
    assert fpga.check_json_main([str(bad)]) == 1
    assert fpga.check_json_main([str(tmp_path / "missing.json")]) == 1


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
        no_fit = (out / f"{name}.no-fit").exists()
        failed = (out / f"{name}.compile-error").exists()
        # Exactly one outcome per size, and the zoo's own slp never fails to compile.
        assert [compiled, no_fit, failed].count(True) == 1, name
        assert not failed, (out / f"{name}.compile-error").read_text()
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
    sizes = {size for _, size in _listed(out)}
    assert first not in sizes
    assert sizes == set(_slp_sizes()) - {first}


def test_compile_rejects_a_skip_naming_nothing(tmp_path, hw_config, capsys):
    out = tmp_path / "programs"
    argv = _compile_argv(hw_config, out) + _skip_all_but("slp")
    with pytest.raises(SystemExit) as exit:
        fpga.compile_main(argv + ["--skip", "slp-no-such-size"])
    assert exit.value.code != 0
    assert "slp-no-such-size" in capsys.readouterr().err
    # Nothing was compiled.
    assert not (out / "sizes.txt").exists()


def test_compile_leaves_out_experimental_sizes(tmp_path, hw_config):
    experimental = {
        (model, size["name"])
        for model, entry in catalog.listing().items()
        for size in entry["sizes"]
        if size["experimental"] is not None
    }
    if not experimental:
        pytest.skip("no model has an experimental size at the moment")
    # Compile only the models that have one, so the test stays cheap.
    with_experimental = {model for model, _ in experimental}
    skips = [
        arg
        for model in catalog.models()
        if model not in with_experimental
        for arg in ("--skip", model)
    ]
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + skips) == 0
    assert not experimental & {tuple(line) for line in _listed(out)}


def test_compile_includes_experimental_models(tmp_path, hw_config):
    """As the VM benchmarks: an experimental model is compiled, only its
    experimental sizes are left out."""
    listing = catalog.listing()
    experimental = [m for m, entry in listing.items() if entry["experimental"]]
    if not experimental:
        pytest.skip("no model is experimental at the moment")
    skips = [
        arg
        for model in catalog.models()
        if model not in experimental
        for arg in ("--skip", model)
    ]
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + skips) == 0
    listed = {tuple(line) for line in _listed(out)}
    for model in experimental:
        named = catalog.named_sizes(model)
        expected = {
            (model, size["name"])
            for size in listing[model]["sizes"]
            if size["experimental"] is None and vm.runs_on(named[size["name"]], "V80")
        }
        assert expected and expected <= listed


def test_compile_notes_a_hw_config_that_is_not_the_zoo_config(
    tmp_path, hw_config, capsys
):
    loaded = vc.Config.load(str(hw_config))
    loaded.num_cores = 3
    odd = _save_config(loaded, tmp_path / "odd.json")
    skips = [arg for model in catalog.models() for arg in ("--skip", model)]
    assert fpga.compile_main(_compile_argv(odd, tmp_path / "out") + skips) == 0
    assert "isn't the zoo's V80" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error, outcome",
    [
        (vc.AllocationError, "AllocationError"),
        (vc.SaveError, "SaveError"),
    ],
)
def test_compile_records_a_size_that_does_not_fit(
    error, outcome, tmp_path, hw_config, monkeypatch
):
    def too_big(zoo, config):
        raise error("too big")

    monkeypatch.setattr(vm, "compile_instance", too_big)
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    for model, size in _listed(out):
        name = f"{model}-{size}"
        assert (out / f"{name}.no-fit").read_text().strip() == outcome
        assert not (out / f"{name}.vollo").exists()

    runs = tmp_path / "runs"
    runs.mkdir()
    data = fpga.collect(out, runs, "V80", hw_config, VERSION)
    entries = data[VERSION]["slp"]["V80"]
    assert len(entries) == len(_slp_sizes())
    assert all(entry in ({"AllocationError": 0}, {"SaveError": 1}) for entry in entries)


def test_compile_records_any_other_failure(tmp_path, hw_config, monkeypatch):
    def broken(zoo, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(vm, "compile_instance", broken)
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    runs = tmp_path / "runs"
    runs.mkdir()
    data = fpga.collect(out, runs, "V80", hw_config, VERSION)
    entries = data[VERSION]["slp"]["V80"]
    assert entries == [{"CompileError": "RuntimeError: boom"}] * len(_slp_sizes())


def test_compile_does_not_hide_a_zoo_bug_as_a_compile_error(
    tmp_path, hw_config, monkeypatch
):
    def bug(zoo, config):
        raise AssertionError("Unsupported zoo arguments: ['bogus']")

    monkeypatch.setattr(vm, "compile_instance", bug)
    out = tmp_path / "programs"
    with pytest.raises(AssertionError, match="bogus"):
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


def test_compile_instance_matches_vollo_info(hw_config):
    """The saved program is the one the zoo measures."""
    model = "slp"
    name = _slp_sizes()[0]
    program = vm.compile_instance(catalog.build(model, name, "V80"), vm.CONFIGS["V80"])
    result = vm.vollo_info(**catalog.build(model, name, "V80"), config="V80")
    assert isinstance(result, vm.Ok)
    assert program.cycle_count_per_inference() == result.cycle_count


def test_compile_instance_rejects_arguments_it_does_not_compile(hw_config):
    config = vm.CONFIGS["V80"]
    with pytest.raises(AssertionError, match="bogus"):
        vm.compile_instance(dict(bogus=1), config)
    with pytest.raises(AssertionError, match="bogus"):
        vm.compile_instance(dict(entries=[], bogus=1), config)


def test_compile_instance_rejects_several_boards():
    zoo = catalog.build("slp", _slp_sizes()[0], "V80")
    with pytest.raises(AssertionError, match="4 boards"):
        vm.compile_instance({**zoo, "boards": 4}, vm.CONFIGS["V80"])
    # One board is the config itself.
    vm.compile_instance({**zoo, "boards": 1}, vm.CONFIGS["V80"])


# Collecting


def _write_run(runs, name, index, mean, median, p99, throughput):
    """What `vollo-example --json` prints, as `fpga.py` reads it."""
    metrics = dict(latency_us=dict(mean=mean, median=median, p99=p99))
    (runs / f"{name}.{index}.raw.json").write_text(
        json.dumps(dict(metrics=dict(metrics, throughput=throughput)))
    )
    (runs / f"{name}.{index}.spaced.json").write_text(
        json.dumps(dict(metrics=dict(metrics, throughput=throughput / 10)))
    )


@pytest.fixture
def fake_results(tmp_path):
    """One of every outcome a size can have, as the script leaves them."""
    programs, runs = tmp_path / "programs", tmp_path / "runs"
    programs.mkdir()
    runs.mkdir()
    hw = {"name": "card", "clock_mhz": 300.0}
    hw_path = tmp_path / "hw_config.json"
    hw_path.write_text(json.dumps(hw))

    info = dict(param_count=10, cycle_count=100, meta={"width": 4})
    sizes = [
        ("m", "ok"),
        ("m", "norun"),
        ("m", "nofit"),
        ("m", "nosave"),
        ("m", "broken"),
        ("m", "rawfail"),
        ("m", "spacedfail"),
        ("multi", "two"),
    ]
    (programs / "sizes.txt").write_text("".join(f"{m} {s}\n" for m, s in sizes))
    (programs / "m-ok.meta.json").write_text(json.dumps([info]))
    (programs / "m-norun.meta.json").write_text(json.dumps([info]))
    (programs / "m-nofit.no-fit").write_text("AllocationError\n")
    (programs / "m-nosave.no-fit").write_text("SaveError\n")
    (programs / "m-broken.compile-error").write_text(
        "Traceback (most recent call last):\n  ...\nValueError: unsupported op\n"
    )
    (programs / "multi-two.meta.json").write_text(
        json.dumps([dict(info, meta={"prog": "a"}), dict(info, meta={"prog": "b"})])
    )

    _write_run(runs, "m-ok", 0, mean=1.5, median=1.4, p99=3.0, throughput=500000.0)
    # One run of the two failed, as the script leaves it: the failed run's JSON
    # may be there too, but its run-error wins.
    for name, failed in (("m-rawfail", "raw"), ("m-spacedfail", "spaced")):
        (programs / f"{name}.meta.json").write_text(json.dumps([info]))
        _write_run(runs, name, 0, mean=1.5, median=1.4, p99=3.0, throughput=500000.0)
        (runs / f"{name}.0.{failed}.run-error").write_text(
            "vollo-example exited with 1\n"
        )
    (runs / "multi-two.0.raw.run-error").write_text("timed out after 600s\n")
    (runs / "multi-two.1.skipped").write_text("an earlier run timed out\n")
    return programs, runs, hw_path, hw


def test_collect_every_outcome(fake_results):
    programs, runs, hw_path, hw = fake_results
    data = fpga.collect(programs, runs, "V80", hw_path, VERSION)
    assert list(data) == [VERSION]
    ok, norun, nofit, nosave, broken, rawfail, spacedfail = data[VERSION]["m"]["V80"]

    assert ok["Ok"]["config"] == "V80"
    assert ok["Ok"]["param_count"] == 10
    assert ok["Ok"]["cycle_count"] == 100
    assert ok["Ok"]["meta"] == {"width": 4}
    # Spaced is the mean of the spaced run; contiguous is 1 / raw throughput.
    assert ok["Ok"]["latency_spaced"] == {"microseconds": 1.5}
    assert ok["Ok"]["latency_contiguous"] == {"microseconds": 2.0}
    hardware = ok["Ok"]["hardware"]
    assert hardware["name"] == "m-ok"
    assert hardware["model_index"] == 0
    assert hardware["hw_config"] == hw
    assert hardware["raw_buffer"] == dict(mean=1.5, median=1.4, p99=3.0)
    assert hardware["spaced"] == dict(mean=1.5, median=1.4, p99=3.0)

    assert norun == {"RunError": "no run result"}
    assert nofit == {"AllocationError": 0}
    assert nosave == {"SaveError": 1}
    assert broken == {"CompileError": "ValueError: unsupported op"}

    # One run failed: still Ok, with that run's latency null and its error in
    # `hardware`, and the other run's numbers kept.
    stats = dict(mean=1.5, median=1.4, p99=3.0)
    error = {"RunError": "vollo-example exited with 1"}
    assert rawfail["Ok"]["latency_contiguous"] is None
    assert rawfail["Ok"]["latency_spaced"] == {"microseconds": 1.5}
    assert rawfail["Ok"]["hardware"]["raw_buffer"] == error
    assert rawfail["Ok"]["hardware"]["spaced"] == stats
    assert spacedfail["Ok"]["latency_spaced"] is None
    assert spacedfail["Ok"]["latency_contiguous"] == {"microseconds": 2.0}
    assert spacedfail["Ok"]["hardware"]["raw_buffer"] == stats
    assert spacedfail["Ok"]["hardware"]["spaced"] == error

    # One entry per model of a multi-model program, each with its own outcome.
    assert data[VERSION]["multi"]["V80"] == [
        {"RunError": "timed out after 600s"},
        {"Skipped": "an earlier run timed out"},
    ]


def test_collect_main_writes_the_results_json(fake_results, tmp_path):
    programs, runs, hw_path, _ = fake_results
    out = tmp_path / "results.json"
    argv = [
        *("--config", "V80"),
        *("--hw-config", str(hw_path)),
        *("--programs", str(programs)),
        *("--runs", str(runs)),
        *("--out", str(out)),
    ]
    assert fpga.collect_main(argv) == 0
    # The shape of benchmarks/*.json: {version: {model: {config: [entry, ...]}}}.
    data = json.loads(out.read_text())
    assert list(data) == [benchmark_version()]
    assert set(data[benchmark_version()]) == {"m", "multi"}


def test_compile_then_collect_without_a_card(tmp_path, hw_config):
    """The pieces fit: every compiled size becomes an entry a run error can fill."""
    out = tmp_path / "programs"
    assert fpga.compile_main(_compile_argv(hw_config, out) + _skip_all_but("slp")) == 0

    runs = tmp_path / "runs"
    runs.mkdir()
    for model, size in _listed(out):
        name = f"{model}-{size}"
        if (out / f"{name}.vollo").exists():
            (runs / f"{name}.0.raw.run-error").write_text("no card")

    data = fpga.collect(out, runs, "V80", hw_config, VERSION)
    entries = data[VERSION]["slp"]["V80"]
    assert len(entries) == len(_slp_sizes())
    assert {"RunError": "no card"} in entries
    json.dumps(data)
