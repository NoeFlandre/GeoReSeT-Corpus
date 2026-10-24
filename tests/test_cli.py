from typer.testing import CliRunner

from georeset import cli, hub
from georeset.build import stage
from georeset.publish import report

runner = CliRunner()


def test_stems_default_lists_polygon_files(monkeypatch):
    files = frozenset({"polygons/b-latest.parquet", "polygons/a-latest.parquet", "x"})
    monkeypatch.setattr(stage.hub, "revision", {cli.wiki.SOURCE.repo: "r"}.__getitem__)
    monkeypatch.setattr(
        stage.hub, "list_files", lambda repo, rev: {(cli.wiki.SOURCE.repo, "r"): files}[(repo, rev)]
    )
    assert cli._stems(None) == ["a-latest", "b-latest"]
    assert cli._stems(["z-latest"]) == ["z-latest"]


def test_run_chains_every_step(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage, "tasks", lambda sources, stems: calls.append(("tasks", stems)) or [])
    monkeypatch.setattr(stage, "stage", lambda todo, work, d, t: calls.append(("stage", d, t)))
    for name in ("build_meta", "build_shards"):
        monkeypatch.setattr(cli.pipeline, name, lambda *a, name=name: calls.append((name,)))
    monkeypatch.setattr(report, "build_card", lambda *a: calls.append(("card",)) or "README.md")
    monkeypatch.setattr(hub, "publish", lambda out, repo: calls.append(("publish", repo)) or "url")
    args = ["run", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"), "--region", "x-latest"]
    result = runner.invoke(cli.app, [*args, "--publish", "--downloads", "5", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert calls == [
        ("tasks", ["x-latest"]),
        ("stage", 5, 2),
        ("build_meta",),
        ("build_shards",),
        ("card",),
        ("publish", cli.REPO_ID),
    ]


def test_run_without_publish(tmp_path, monkeypatch):
    published = []
    monkeypatch.setattr(stage, "tasks", lambda *a: [])
    monkeypatch.setattr(stage, "stage", lambda *a: None)
    monkeypatch.setattr(cli.pipeline, "build_meta", lambda *a: None)
    monkeypatch.setattr(cli.pipeline, "build_shards", lambda *a: None)
    monkeypatch.setattr(report, "build_card", lambda *a: "README.md")
    monkeypatch.setattr(hub, "publish", lambda *a: published.append(a))
    result = runner.invoke(cli.app, ["run", "--work", str(tmp_path), "--region", "x"])
    assert result.exit_code == 0, result.output
    assert published == []
