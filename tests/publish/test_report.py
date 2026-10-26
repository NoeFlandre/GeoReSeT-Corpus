import json

import shapely
from conftest import build_all
from huggingface_hub import DatasetCard

from georeset.publish import geo, report


def test_build_card(layout, monkeypatch):
    caches = []
    monkeypatch.setattr(
        geo, "load_land", lambda cache: caches.append(cache) or [shapely.box(-10, 35, 30, 60)]
    )
    build_all(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    readme = report.build_card(layout, "me/data", "https://github.com/me/code")
    assert readme == layout.out / "README.md"
    assert caches == [layout.work / "cache"]
    text = readme.read_text()
    assert 'load_dataset("me/data"' in text
    assert "Code: https://github.com/me/code" in text
    assert DatasetCard(text).data.pretty_name == "GeoReSeT Corpus"
    assert "**4 sentences in 2 languages, tied to 3 OpenStreetMap polygons in 2 regions" in text
    assert (layout.out / "assets" / "map.png").exists()
    assert (layout.out / "assets" / "languages.png").exists()
    assert (layout.out / "assets" / "hero.png").read_bytes()[:4] == b"\x89PNG"
    assert text.index("assets/hero.png") < text.index("# GeoReSeT Corpus")
    assert report.partitions(layout.out) == [("website", "eng_Latn"), ("wikipedia", "fra_Latn")]


def test_build_fused_card(layout, raw_scenes, monkeypatch):
    from georeset.build import fuse

    monkeypatch.setattr(geo, "load_land", lambda cache: [shapely.box(-10, 35, 30, 60)])
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
    hero = (layout.out / "assets" / "hero.png").read_bytes()
    assert hero == report.FUSED_HERO.read_bytes() != report.HERO.read_bytes()


def test_plain_card_uses_plain_hero(layout, monkeypatch):
    monkeypatch.setattr(geo, "load_land", lambda cache: [shapely.box(-10, 35, 30, 60)])
    build_all(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    report.build_card(layout, "me/data", "https://github.com/me/code")
    assert (layout.out / "assets" / "hero.png").read_bytes() == report.HERO.read_bytes()
    assert report.scene_context(layout.out, layout.work) is None
