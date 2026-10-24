import polars as pl
import shapely
from conftest import SQUARE, labels_columns

from georeset.sources import website


def _website(write, labels_rows):
    polygons = write(
        "polygons",
        {
            "polygon_id": ["r:way/2", "r:way/1", "r:node/3"],
            "osm_type": ["way", "way", "node"],
            "osm_id": [2, 1, 3],
            "name": ["B", "A", "C"],
            "tags": ['{"shop":"bakery"}', '{"amenity":"cafe"}', "{}"],
            "lat": [2.0, 1.0, 3.0],
            "lon": [5.0, 4.0, 6.0],
            "area_m2": [20.0, 10.0, 30.0],
            "geometry": [SQUARE, SQUARE, SQUARE],
            "website": ["http://x", "http://x", None],
            "contact_website": [None, None, "http://y"],
            "website_sentences": [["Other copy"], ["Hello there", "Second one"], None],
            "contact_website_sentences": [None, None, None],
            "website_text": ["Other copy", "Hello there Second one", None],
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
    assert staged.sentences.sort("doc_key", "position").rows() == [
        ("website", "http://x", 0, "Hello there", "yes", None, "eng_Latn", "http://x"),
        ("website", "http://x", 1, "Second one", "failed", "truncated", "eng_Latn", "http://x"),
        (
            "website",
            "http://y",
            0,
            "Whole page text",
            "skipped_unsplit",
            "unsplit_upstream",
            "vec_Latn",
            "http://y",
        ),
    ]
    assert staged.links.sort("polygon_id").rows() == [
        ("http://y", "node/3"),
        ("http://x", "way/1"),
        ("http://x", "way/2"),
    ]
    p = staged.polygons.sort("polygon_id")
    assert p.drop("geometry").rows() == [
        ("node/3", "C", None, 3.0, 6.0, 30.0),
        ("way/1", "A", "amenity=cafe", 1.0, 4.0, 10.0),
        ("way/2", "B", "shop=bakery", 2.0, 5.0, 20.0),
    ]
    assert p.columns[-1] == "geometry"
    assert [shapely.from_wkb(g).area for g in p["geometry"]] == [1.0, 1.0, 1.0]


def test_website_files():
    assert website.files("x-latest") == {
        "polygons": "polygons/x-latest.parquet",
        "labels": "labels/polygons/x-latest.parquet",
    }
