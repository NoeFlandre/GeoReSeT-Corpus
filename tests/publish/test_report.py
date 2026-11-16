import json

from conftest import build_all
from huggingface_hub import DatasetCard

from georeset.publish import report


def test_build_card(layout):
    build_all(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    readme = report.build_card(layout, "me/data", "https://github.com/me/code")
    assert readme == layout.out / "README.md"
    text = readme.read_text()
    assert 'load_dataset("me/data"' in text
    assert "Code: https://github.com/me/code" in text
    assert DatasetCard(text).data.pretty_name == "GeoReSeT Corpus"
    assert "**4 sentences in 2 languages, tied to 3 OpenStreetMap polygons in 2 regions" in text
    for name in ["map.png", *report.SENTENCE_MAPS]:
        assert (layout.out / "assets" / name).exists()
    assert not (layout.out / "assets" / "map_scenes.png").exists()
    assert (layout.out / "assets" / "languages.png").exists()
    assert (layout.out / "assets" / "hero.png").read_bytes()[:4] == b"\x89PNG"
    assert text.index("assets/hero.png") < text.index("# GeoReSeT Corpus")
    assert report.partitions(layout.out) == [("website", "eng_Latn"), ("wikipedia", "fra_Latn")]


def test_build_fused_card(layout, raw_scenes):
    from georeset.build import fuse

    build_all(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    (layout.work / "scenes.json").write_text(json.dumps({"me/raw": "f" * 40}))
    fuse.build_scenes(raw_scenes, layout.out)
    text = report.build_card(layout, "me/data", "https://github.com/me/code").read_text()
    assert DatasetCard(text).data.pretty_name == "GeoReSeT Corpus Fused"
    assert "**Plus 2 English scene descriptions" in text
    assert "tied to the locations of the 3 geotagged photos they describe" in text
    assert "| Transportation network | 1 | 2 |" in text
    assert "| mapillary | 1 | 100.0% |" in text
    assert "`me/raw` @ `fffffff`" in text
    assert (layout.out / "assets" / "map_scenes.png").exists()
    assert "assets/map_scenes.png" in text
    assert "all sentences and all scene photos" in text
    assert report.scene_points(layout.out)["n"].sum() == 3
    hero = (layout.out / "assets" / "hero.png").read_bytes()
    assert hero == report.FUSED_HERO.read_bytes() != report.HERO.read_bytes()


def test_plain_card_uses_plain_hero(layout):
    build_all(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    report.build_card(layout, "me/data", "https://github.com/me/code")
    assert (layout.out / "assets" / "hero.png").read_bytes() == report.HERO.read_bytes()
    assert report.scene_context(layout.out, layout.work) is None
    assert report.scene_points(layout.out) is None


def test_theme_context_renders_each_computed_pca(tmp_path, monkeypatch):
    assert report.theme_context(tmp_path, tmp_path / "assets") == ()
    (tmp_path / "themes" / "scenes").mkdir(parents=True)
    (tmp_path / "themes" / "scenes" / "summary.json").write_text("{}")
    monkeypatch.setattr(report.themes, "render", lambda work, assets, corpus: (work, assets, corpus.name))
    assert report.theme_context(tmp_path, tmp_path / "assets") == ((tmp_path, tmp_path / "assets", "scenes"),)


def _translated(out, folder, rows):
    import polars as pl

    path = out / folder / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "translation": pl.String,
            "translation_language": pl.String,
            "translation_flags": pl.List(pl.String),
        },
        orient="row",
    ).write_parquet(path)


def test_translation_context_reads_the_summary_and_the_shards(tmp_path):
    out, work = tmp_path / "out", tmp_path / "work"
    assert report.translation_context(out, work) is None
    _translated(
        out, "data/wikipedia/fra_Latn", [("nllb", "fra_Latn", ["truncated"]), ("nllb", "fra_Latn", [])]
    )
    _translated(out, "unlabelled", [("original", None, []), ("nllb", "deu_Latn", ["empty", "truncated"])])
    work.mkdir()
    meta = {"distinct": 2, "model": "m/x", "revision": "r", "threshold": 0.9, "release": "s"}
    (work / "translation.json").write_text(json.dumps(meta))
    context = report.translation_context(out, work)
    assert context is not None
    assert context.kinds.rows() == [("nllb", 3), ("original", 1)]
    assert context.flags.rows() == [("truncated", 2, 2 / 3), ("empty", 1, 1 / 3)]
    assert (context.distinct, context.languages, context.model, context.revision, context.threshold) == (
        2,
        2,
        "m/x",
        "r",
        0.9,
    )


def test_translation_context_reads_its_files_by_their_exact_names(tmp_path, monkeypatch):
    import polars as pl

    names = []

    class Seen(type(tmp_path)):
        def __truediv__(self, name):
            names.append(name)
            return super().__truediv__(name)

    out, work = tmp_path / "out", tmp_path / "work"
    _translated(out, "unlabelled", [("nllb", "deu_Latn", [])])
    _translated(out, "data/wikipedia/fra_Latn", [("nllb", "fra_Latn", [])])
    work.mkdir()
    meta = {"distinct": 1, "model": "m/x", "revision": "r", "threshold": 0.9, "release": "s"}
    (work / "translation.json").write_text(json.dumps(meta))
    scanned, scan = [], pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda path: scanned.append(path) or scan(path))
    assert report.translation_context(out, Seen(work)) is not None
    assert names == ["translation.json"]
    assert scanned == [out / "data" / "*" / "*" / "*.parquet", out / "unlabelled" / "*.parquet"]
