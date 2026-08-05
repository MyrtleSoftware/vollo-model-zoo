"""
The identity of a benchmark run: the Vollo SDK it was measured against *and*
the version of this repo that defined the models. Both matter, because a change
to a model file moves the numbers just as a new compiler does.

The two are combined into a single PEP 440 version string, `<vollo>+zoo.<zoo>`,
so `packaging.version.parse` still orders benchmark files. The zoo segments are
numeric, so they order chronologically rather than lexically:

    26.1.0 < 28.1.0 < 28.1.0+zoo.0.2.0 < 28.1.0+zoo.0.10.0 < 28.2.0+zoo.0.2.0
"""

import importlib.metadata

from beartype import beartype
from beartype.typing import Optional

_ZOO_TAG = "zoo"

UNKNOWN = "unknown"


@beartype
def _installed(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return UNKNOWN


@beartype
def vollo_version() -> str:
    """
    Version of the installed Vollo compiler that models are measured against.
    """
    return _installed("vollo-compiler")


@beartype
def zoo_version() -> str:
    """
    Version of this repo, i.e. `[project] version` in `pyproject.toml`.

    Bump it whenever a change alters what the benchmarks measure; the weekly
    benchmark workflow has no other way to tell that a re-run is due.
    """
    return _installed("vollo-model-zoo")


@beartype
def benchmark_version(vollo: Optional[str] = None, zoo: Optional[str] = None) -> str:
    """
    The combined `<vollo>+zoo.<zoo>` version identifying a benchmark run.

    Both parts default to what is installed. The benchmark workflow builds the
    same string in shell, reading the zoo version from `pyproject.toml`; keep
    the two in step.
    """
    vollo = vollo_version() if vollo is None else vollo
    zoo = zoo_version() if zoo is None else zoo

    return f"{vollo}+{_ZOO_TAG}.{zoo}"


@beartype
def split_benchmark_version(version: str) -> tuple[str, Optional[str]]:
    """
    Inverse of `benchmark_version`. The zoo version is `None` for benchmark
    files predating it, which recorded the Vollo version alone.
    """
    vollo, _, zoo = version.partition(f"+{_ZOO_TAG}.")

    return vollo, zoo or None


@beartype
def describe_version(version: str) -> str:
    """
    A benchmark version rendered for a human, in logs and reports.
    """
    vollo, zoo = split_benchmark_version(version)

    if zoo is None:
        return f"Vollo SDK {vollo}"

    return f"Vollo SDK {vollo}, model zoo {zoo}"
