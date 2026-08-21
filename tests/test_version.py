import tomllib
from pathlib import Path

from beartype import beartype
from packaging.version import parse as parse_version

from vollo_model_zoo.version import (
    benchmark_version,
    describe_version,
    split_benchmark_version,
    zoo_version,
)


@beartype
def test_round_trip():
    version = benchmark_version("28.1.0", "0.2.0")

    assert version == "28.1.0+zoo.0.2.0"
    assert split_benchmark_version(version) == ("28.1.0", "0.2.0")


@beartype
def test_benchmarks_predating_the_zoo_version():
    assert split_benchmark_version("26.1.0") == ("26.1.0", None)
    assert describe_version("26.1.0") == "Vollo SDK 26.1.0"


@beartype
def test_versions_sort_chronologically():
    """
    `plot.py` orders its x axis with `packaging.version.parse`, so a benchmark
    version has to be a PEP 440 version whose ordering is the order the runs
    happened in: SDK first, then zoo, numerically (0.10.0 after 0.2.0).
    """
    versions = [
        "26.1.0",
        "28.1.0",
        benchmark_version("28.1.0", "0.1.0"),
        benchmark_version("28.1.0", "0.2.0"),
        benchmark_version("28.1.0", "0.10.0"),
        benchmark_version("28.2.0", "0.2.0"),
    ]

    assert sorted(versions, key=parse_version) == versions


@beartype
def test_installed_zoo_version_matches_pyproject():
    """
    The benchmark workflow reads the zoo version from `pyproject.toml` while
    `benchmark` reads it from the installed package; they have to agree or the
    two would name different files. A failure here is usually a stale venv.
    """
    pyproject = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject, "rb") as f:
        expected = tomllib.load(f)["project"]["version"]

    assert zoo_version() == expected, "stale venv? re-run without `--no-sync`"
