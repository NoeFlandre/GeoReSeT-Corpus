from typer.testing import CliRunner

from georeset import cli
from georeset.label import jobs

runner = CliRunner()
EMPTY_REPORT = {"rows": 0, "matches": 0, "failed": 0, "mismatches": []}


def test_label_commands_call_their_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "plan_corpus", lambda *args: calls.append(("plan", *args)))
    monkeypatch.setattr(jobs, "work_on_gpu", lambda *args: calls.append(("work", *args)))
    monkeypatch.setattr(jobs, "report", lambda *args: EMPTY_REPORT)
    monkeypatch.setattr(jobs, "remove", lambda *args: calls.append(("remove", *args)))
    options = ["--bucket", "me/b"]
    assert runner.invoke(cli.app, ["label-plan", *options, "--sample", "8"]).exit_code == 0
    assert runner.invoke(cli.app, ["label-work", *options]).exit_code == 0
    assert runner.invoke(cli.app, ["label-report", *options]).exit_code == 0
    assert runner.invoke(cli.app, ["label-remove", *options]).exit_code == 0
    assert [call[0] for call in calls] == ["plan", "work", "remove"]
    _, bucket, _, sample, seed = calls[0]
    assert (bucket.id, sample, seed) == ("me/b", 8, 0)
    assert calls[1][1].id == "me/b"
    assert calls[2][1].id == "me/b"


def test_label_report_prints_the_summary(monkeypatch):
    monkeypatch.setattr(jobs, "report", lambda *args: EMPTY_REPORT)
    result = runner.invoke(cli.app, ["label-report", "--bucket", "me/b"])
    assert '"rows": 0' in result.output
