import json

import pytest

from tools.crap import Function, crap, functions, report


def test_crap_formula():
    assert crap(complexity=1, coverage=1.0) == 1
    assert crap(complexity=3, coverage=0.0) == 12
    assert crap(complexity=2, coverage=0.5) == pytest.approx(2 + 4 * 0.125)


def test_functions_join_complexity_and_coverage(tmp_path):
    source = tmp_path / "m.py"
    source.write_text("def f(x):\n    if x:\n        return 1\n    return 2\n\n\ndef g():\n    return 3\n")
    cov = {"files": {str(source): {"executed_lines": [1, 2, 4, 7], "missing_lines": [3, 8]}}}
    out = sorted(functions(cov), key=lambda f: f.name)
    assert out == [
        Function(str(source), "f", 1, 2, 3 / 4),
        Function(str(source), "g", 7, 1, 1 / 2),
    ]


def test_report_flags_functions_at_or_over_threshold(tmp_path):
    fns = [Function("a.py", "ok", 1, 1, 1.0), Function("a.py", "bad", 5, 3, 0.0)]
    lines, worst = report(fns, threshold=6)
    assert worst == 12
    assert lines == ["CRAP  12.0  a.py:5 bad"]


def test_main_exit_code(tmp_path, monkeypatch, capsys):
    from tools import crap as module

    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps({"files": {}}))
    assert module.main([str(cov_file), "6"]) == 0
    assert "max CRAP 0.0" in capsys.readouterr().out
    monkeypatch.setattr(module, "functions", lambda cov: [Function("a.py", "bad", 1, 9, 1.0)])
    assert module.main([str(cov_file), "6"]) == 1
