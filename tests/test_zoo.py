import re
import sys

from vollo_model_zoo import zoo


def _zoo(monkeypatch, capsys, *args: str) -> str:
    monkeypatch.setattr(sys, "argv", ["zoo", *args])
    assert zoo.main() == 0
    return capsys.readouterr().out


def test_markdown(monkeypatch, capsys):
    """
    `zoo <model> --markdown` prints a markdown table, to paste into issues and
    PRs: a header, a separator, then one row per size, all with the same cells.
    """
    out = _zoo(monkeypatch, capsys, "mlp", "--markdown")
    table = [line for line in out.splitlines() if line.startswith("|")]
    header, separator, *rows = table
    assert "compute latency (us)" in header
    assert re.fullmatch(r"\|( *:?-+:? *\|)+", separator)
    assert len(rows) > 0
    assert len({line.count("|") for line in table}) == 1


def test_plain_by_default(monkeypatch, capsys):
    out = _zoo(monkeypatch, capsys, "mlp")
    assert "compute latency (us)" in out
    assert "|" not in out
