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


def test_themes_sentences(tmp_path):
    import polars as pl

    data = tmp_path / "data.parquet"
    pl.DataFrame(
        {"id": ["a"], "text": ["x"], "landuse": [True], "language": ["fra_Latn"], "lat": [1.0], "lon": [2.0]}
    ).write_parquet(data)
    result = runner.invoke(
        cli.app, ["themes-sentences", "--sentences", str(tmp_path / "s"), "--data", str(data)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "1"
    assert len(list((tmp_path / "s").glob("train-*.parquet"))) == cli.themes.LANDUSE.shards


def test_themes_sentences_reads_the_corpus_from_the_hub(tmp_path, monkeypatch):
    import polars as pl

    seen = []
    scenes = pl.LazyFrame({"scene_id": [1], "text": ["x"], "lat": [1.0], "lon": [2.0]})
    monkeypatch.setattr(cli.pl, "scan_parquet", lambda source: seen.append(source) or scenes)
    result = runner.invoke(cli.app, ["themes-sentences", "--corpus", "scenes", "--sentences", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert seen == [f"hf://datasets/{cli.FUSED_REPO_ID}/scenes/*.parquet"]
    assert [p.name for p in tmp_path.glob("*.parquet")] == ["train-00000-of-00001.parquet"]
    assert runner.invoke(cli.app, ["themes-sentences", "--corpus", "nope"]).exit_code == 2


def test_themes_downloads_embeddings_then_computes(tmp_path, monkeypatch):
    for name in ("train-00001-of-00002.parquet", "train-00000-of-00002.parquet"):
        (tmp_path / name).touch()
    seen = {}
    monkeypatch.setattr(hub, "revision", lambda repo: "rev")
    monkeypatch.setattr(
        hub, "snapshot", lambda repo, rev, dest, patterns: seen.update(get=(rev, dest, patterns))
    )
    monkeypatch.setattr(cli.pl, "scan_parquet", lambda glob: f"scan {glob}")
    monkeypatch.setattr(cli.stats, "photo_hexagons", lambda photos: f"places from {photos}")

    def compute(sentences, embeddings, work, corpus, places):
        seen.update(
            sentences=[p.name for p in sentences],
            embeddings=embeddings,
            work=work,
            corpus=corpus,
            places=places,
        )
        return work / "themes"

    monkeypatch.setattr(cli.themes, "compute", compute)
    cache, work = tmp_path / "c", tmp_path / "w"
    args = [
        "themes",
        "--work",
        str(work),
        "--sentences",
        str(tmp_path),
        "--cache",
        str(cache),
        "--repo",
        "me/x",
        "--corpus",
        "scenes",
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seen["get"] == ("rev", cache, ["scene_embeddings/*"])
    assert seen["places"] == "places from scan hf://datasets/me/x@rev/scene_photos/*.parquet"
    assert seen["sentences"] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert seen["embeddings"] == [cache / "scene_embeddings" / n for n in seen["sentences"]]
    assert seen["corpus"] is cli.themes.SCENES
    assert seen["work"] == work
    # The sentences count at their own location.
    assert runner.invoke(cli.app, args[:-2]).exit_code == 0
    assert seen["corpus"] is cli.themes.LANDUSE and seen["places"] is None
