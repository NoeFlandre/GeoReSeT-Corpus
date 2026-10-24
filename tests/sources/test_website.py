import polars as pl
import shapely
from conftest import SQUARE, labels_columns

from georeset.sources import website


def _website(write, labels_rows):
    polygons = write(
        "polygons",
        {
            "polygon_id": ["r:way/1", "r:way/2", "r:node/3"],
            "osm_type": ["way", "way", "node"],
            "osm_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "tags": ['{"amenity":"cafe"}', '{"shop":"bakery"}', "{}"],
            "lat": [1.0, 2.0, 3.0],
            "lon": [4.0, 5.0, 6.0],
            "area_m2": [10.0, 20.0, 30.0],
            "geometry": [SQUARE, SQUARE, SQUARE],
            "website": ["http://x", "http://x", None],
            "contact_website": [None, None, "http://y"],
            "website_sentences": [["Hello there", "Second one"], ["Other copy"], None],
            "contact_website_sentences": [None, None, None],
            "website_text": ["Hello there Second one", "Other copy", None],
            "contact_website_text": [None, None, "Whole page text"],
        },
    )
    labels = write(
        "labels",
        labels_columns(labels_rows, ["polygon_id", "field", "sentence_index"]),
        schema={
            "polygon_id": pl.String,
            "field": pl.String,
            "sentence_index": pl.Int32,
            "decision": pl.String,
            "failure_reason": pl.String,
            "language": pl.String,
        },
    )
    return website.transform({"polygons": polygons, "labels": labels})


def test_website_transform(write):
    staged = _website(
        write,
        [
            ("r:way/1", "website", 0, "yes", None, "eng_Latn"),
            ("r:way/1", "website", 1, "failed", "truncated", "eng_Latn"),
            ("r:way/2", "website", 0, "no", None, "eng_Latn"),
            (
                "r:node/3",
                "contact_website",
                0,
                "skipped_unsplit",
                "unsplit_upstream",
                "vec_Latn",
            ),
        ],
    )
    s = staged.sentences.sort("doc_key", "position")
    assert s["text"].to_list() == ["Hello there", "Second one", "Whole page text"]
    assert s["label"].to_list() == ["yes", "failed", "skipped_unsplit"]
    assert s["failure_reason"].to_list() == [None, "truncated", "unsplit_upstream"]
    assert s["doc_key"].to_list() == ["http://x", "http://x", "http://y"]
    assert set(s["source"]) == {"website"}
    assert staged.links.sort("polygon_id").rows() == [
        ("http://y", "node/3"),
        ("http://x", "way/1"),
        ("http://x", "way/2"),
    ]
    p = staged.polygons.sort("polygon_id")
    assert p["osm_tag"].to_list() == [None, "amenity=cafe", "shop=bakery"]
    assert shapely.from_wkb(p["geometry"][0]).area == 1.0
    assert p.columns == [
        "polygon_id",
        "name",
        "osm_tag",
        "lat",
        "lon",
        "area_m2",
        "geometry",
    ]


def test_website_files():
    assert website.files("x-latest") == {
        "polygons": "polygons/x-latest.parquet",
        "labels": "labels/polygons/x-latest.parquet",
    }
