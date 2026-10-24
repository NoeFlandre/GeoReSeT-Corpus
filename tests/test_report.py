import json

import shapely
from huggingface_hub import DatasetCard
from test_pipeline import build

from georeset import plots, report


def test_build_card(layout, monkeypatch):
    monkeypatch.setattr(plots, "load_land", lambda cache: [shapely.box(-10, 35, 30, 60)])
    build(layout)
    (layout.work / "revisions.json").write_text(json.dumps({"me/up": "abcdef123"}))
    readme = report.build_card(layout, "me/data", "https://github.com/me/code")
    text = readme.read_text()
    assert DatasetCard(text).data.pretty_name == "GeoReSeT Corpus"
    assert "**4 sentences in 2 languages, tied to 3 OpenStreetMap polygons in 2 regions" in text
    assert (layout.out / "assets" / "map.png").exists()
    assert (layout.out / "assets" / "languages.png").exists()
    assert report.partitions(layout.out) == [("website", "eng_Latn"), ("wikipedia", "fra_Latn")]
