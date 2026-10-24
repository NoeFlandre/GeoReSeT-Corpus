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
