import re

from vollo_model_zoo import zoo


def test_table_is_markdown(capsys):
    """
    `zoo <model>` prints a markdown table, which people paste into issues and
    PRs: a header, a separator, then one row per size, all with the same cells.
    """
    assert zoo.run_model("mlp", "V80", use_json=False) == 0
    table = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("|")
    ]
    header, separator, *rows = table
    assert "compute latency (us)" in header
    assert re.fullmatch(r"\|( *:?-+:? *\|)+", separator)
    assert len(rows) > 0
    assert len({line.count("|") for line in table}) == 1
