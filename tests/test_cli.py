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
    monkeypatch.setattr(
        hub, "publish", lambda out, repo, private: calls.append(("publish", repo, private)) or "url"
    )
    args = ["run", "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"), "--region", "x-latest"]
    result = runner.invoke(cli.app, [*args, "--publish", "--downloads", "5", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert calls == [
        ("tasks", ["x-latest"]),
        ("stage", 5, 2),
        ("build_meta",),
        ("build_shards",),
        ("card",),
        ("publish", cli.REPO_ID, False),
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


def test_fuse_chains_corpus_scenes_card_and_publish(tmp_path, monkeypatch):
    calls = []
    work, out = tmp_path / "w", tmp_path / "o"
    monkeypatch.setattr(
        cli.fusion, "fetch_corpus", lambda *a: calls.append(("corpus", *a))
    )  # repo, rev, out, work
    monkeypatch.setattr(cli.fusion, "fetch_scenes", lambda w: calls.append(("raw", w)) or w / "raw")
    monkeypatch.setattr(cli.fusion, "build_scenes", lambda raw, o: calls.append(("scenes", raw, o)))
    monkeypatch.setattr(report, "build_card", lambda layout, *a: calls.append(("card", layout, *a)) or "R")
    monkeypatch.setattr(hub, "publish", lambda *a: calls.append(("publish", *a)) or "url")
    result = runner.invoke(
        cli.app, ["fuse", "--work", str(work), "--out", str(out), "--publish", "--private"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        ("corpus", cli.REPO_ID, cli.CORPUS_RELEASE, out, work),
        ("raw", work),
        ("scenes", work / "raw", out),
        ("card", cli.pipeline.Layout(work, out), cli.FUSED_REPO_ID, cli.CODE_URL),
        ("publish", out, cli.FUSED_REPO_ID, True),
    ]


def test_fuse_without_publish_and_other_release(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli.fusion, "fetch_corpus", lambda repo, rev, *a: seen.append(rev))
    monkeypatch.setattr(cli.fusion, "fetch_scenes", lambda w: w)
    monkeypatch.setattr(cli.fusion, "build_scenes", lambda *a: None)
    monkeypatch.setattr(report, "build_card", lambda *a: "R")
    monkeypatch.setattr(hub, "publish", lambda *a: seen.append("published"))
    result = runner.invoke(cli.app, ["fuse", "--work", str(tmp_path), "--corpus", "abc123"])
    assert result.exit_code == 0, result.output
    assert seen == ["abc123"]


def test_fuse_publishes_public_by_default(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli.fusion, "fetch_corpus", lambda *a: None)
    monkeypatch.setattr(cli.fusion, "fetch_scenes", lambda w: w)
    monkeypatch.setattr(cli.fusion, "build_scenes", lambda *a: None)
    monkeypatch.setattr(report, "build_card", lambda *a: "R")
    monkeypatch.setattr(hub, "publish", lambda out, repo, private: seen.append(private) or "url")
    result = runner.invoke(cli.app, ["fuse", "--work", str(tmp_path), "--publish"])
    assert result.exit_code == 0, result.output
    assert seen == [False]
