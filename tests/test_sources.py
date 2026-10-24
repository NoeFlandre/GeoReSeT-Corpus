import polars as pl
import shapely
from conftest import SQUARE

from georeset.sources import (
    SENTENCE_SCHEMA,
    attach_text,
    description,
    finish_sentences,
    website,
    wiki,
)

LABEL_KEYS = {"decision": None, "failure_reason": None, "language": None}


def _labels(rows: list[tuple], keys: list[str]) -> dict:
    names = [*keys, "decision", "failure_reason", "language"]
    return {name: [row[i] for row in rows] for i, name in enumerate(names)}


# ---------- shared helpers ----------


def test_attach_text_picks_sentence_or_full_text():
    docs = pl.DataFrame(
        {
            "k": ["a", "b"],
            "sentences": [["s0", "s1"], None],
            "full_text": [None, "whole"],
        }
    )
    labels = pl.DataFrame(
        {
            "k": ["a", "a", "b"],
            "sentence_index": [1, 0, 0],
            "label": ["yes", "no", "skipped_unsplit"],
        },
        schema_overrides={"sentence_index": pl.Int32},
    )
    out = attach_text(labels, docs, ["k"])
    assert out["text"].to_list() == ["s1", "s0", "whole"]
    assert out.columns == ["k", "sentence_index", "label", "text"]


def test_attach_text_out_of_range_index_is_null():
    docs = pl.DataFrame({"k": ["a"], "sentences": [["s0"]], "full_text": [None]})
    labels = pl.DataFrame(
        {"k": ["a"], "sentence_index": [5], "label": ["yes"]},
        schema_overrides={"sentence_index": pl.Int32},
    )
    assert attach_text(labels, docs, ["k"])["text"].to_list() == [None]


def test_finish_sentences_filters_debris_and_normalizes():
    raw = pl.DataFrame(
        {
            "source": ["s", "s", "s"],
            "doc_key": ["d", "d", "d"],
            "position": [0, 1, 2],
            "text": ["Bonjour", "-", None],
            "label": ["yes", "no", "no"],
            "failure_reason": [None, None, None],
            "language": ["fr", "fr", "fr"],
            "url": ["u", "u", "u"],
            "extra": [1, 2, 3],
        }
    )
    out = finish_sentences(raw, "wikimedia")
    assert out.schema == SENTENCE_SCHEMA
    assert out["text"].to_list() == ["Bonjour"]
    assert out["language"].to_list() == ["fra_Latn"]


# ---------- website ----------


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
        _labels(labels_rows, ["polygon_id", "field", "sentence_index"]),
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


def test_description_files():
    assert description.files("x") == {
        "polygons": "data/x.parquet",
        "texts": "language-v1/data/x.parquet",
        "labels": "labels/language-v1/data/x.parquet",
    }


def test_website_files():
    assert website.files("x-latest") == {
        "polygons": "polygons/x-latest.parquet",
        "labels": "labels/polygons/x-latest.parquet",
    }


# ---------- description ----------


def test_description_transform(write):
    tags_type = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
    polygons = write(
        "data",
        {
            "osm_type": ["way"],
            "osm_id": [7],
            "name": ["Hut"],
            "tags": [[{"key": "tourism", "value": "alpine_hut"}]],
            "area_m2": [5.0],
            "geometry": [shapely.to_wkb(shapely.box(0, 0, 1, 1))],
            "bbox_min_x": [0.0],
            "bbox_min_y": [10.0],
            "bbox_max_x": [2.0],
            "bbox_max_y": [20.0],
        },
        schema={
            "osm_type": pl.String,
            "osm_id": pl.Int64,
            "name": pl.String,
            "tags": tags_type,
            "area_m2": pl.Float64,
            "geometry": pl.Binary,
            "bbox_min_x": pl.Float64,
            "bbox_min_y": pl.Float64,
            "bbox_max_x": pl.Float64,
            "bbox_max_y": pl.Float64,
        },
    )
    texts = write(
        "texts",
        {
            "description_identity": ["id1", "id2"],
            "osm_type": ["way", "way"],
            "osm_id": [7, 7],
            "tag_key": ["description", "description:fr"],
            "original_text": ["Big hut. Nice view", "Grand refuge"],
            "sentences": [["Big hut.", "Nice view"], []],
        },
    )
    labels = write(
        "labels",
        _labels(
            [
                ("id1", "description", 1, "yes", None, "eng"),
                (
                    "id2",
                    "description:fr",
                    0,
                    "skipped_unsplit",
                    "unsplit_upstream",
                    "fra",
                ),
            ],
            ["description_identity", "tag_key", "sentence_index"],
        ),
        schema={
            "description_identity": pl.String,
            "tag_key": pl.String,
            "sentence_index": pl.Int32,
            "decision": pl.String,
            "failure_reason": pl.String,
            "language": pl.String,
        },
    )
    staged = description.transform({"polygons": polygons, "texts": texts, "labels": labels})
    s = staged.sentences.sort("doc_key")
    assert s["text"].to_list() == ["Nice view", "Grand refuge"]
    assert s["language"].to_list() == ["eng_Latn", "fra_Latn"]
    assert s["url"].to_list() == ["https://www.openstreetmap.org/way/7"] * 2
    assert set(s["source"]) == {"osm_description"}
    assert staged.links.sort("doc_key").rows() == [("id1", "way/7"), ("id2", "way/7")]
    p = staged.polygons.row(0, named=True)
    assert (p["polygon_id"], p["osm_tag"], p["lat"], p["lon"]) == (
        "way/7",
        "tourism=alpine_hut",
        15.0,
        1.0,
    )


# ---------- wiki ----------


def _wiki_project(write, project, rows):
    sentences = write(
        f"{project}_sentences",
        {
            "sentence_id": [r[0] for r in rows],
            "document_id": [r[1] for r in rows],
            "section_index": [r[2] for r in rows],
            "sentence_index": [r[3] for r in rows],
            "text": [r[4] for r in rows],
        },
    )
    labels = write(
        f"{project}_labels",
        {
            "sentence_id": [r[0] for r in rows],
            "decision": ["yes"] * len(rows),
            "failure_reason": [None] * len(rows),
            "language": [r[5] for r in rows],
        },
        schema={k: pl.String for k in ["sentence_id", "decision", "failure_reason", "language"]},
    )
    return {f"{project}_sentences": sentences, f"{project}_labels": labels}


def test_wiki_transform_orders_sections_and_builds_permalinks(write):
    paths = {
        "polygons": write(
            "polygons",
            {
                "osm_type": ["relation"],
                "osm_id": [9],
                "name": ["Forest"],
                "tags": ['{"landuse":"forest"}'],
                "lat": [1.0],
                "lon": [2.0],
                "area_m2": [3.0],
                "geometry": [SQUARE],
            },
        ),
        "links": write(
            "links",
            {
                "osm_type": ["relation"],
                "osm_id": [9],
                "document_id": ["Q1:wikipedia:fr:10:99"],
            },
        ),
    }
    doc = "Q1:wikipedia:fr:10:99"
    paths |= _wiki_project(
        write,
        "wikipedia",
        [
            ("c", doc, 1, 0, "Third part", "fr"),
            ("a", doc, 0, 0, "First part", "fr"),
            ("b", doc, 0, 1, "Second part", "fr"),
        ],
    )
    staged = wiki.transform(paths)
    s = staged.sentences.sort("position")
    assert s["text"].to_list() == ["First part", "Second part", "Third part"]
    assert s["position"].to_list() == [0, 1, 2]
    assert set(s["url"]) == {"https://fr.wikipedia.org/w/index.php?oldid=99"}
    assert set(s["source"]) == {"wikipedia"}
    assert staged.links.rows() == [(doc, "relation/9")]
    assert staged.polygons["osm_tag"].to_list() == ["landuse=forest"]


def test_wiki_files_lists_both_projects():
    files = wiki.files("x")
    assert files["wikivoyage_labels"] == "labels/wikivoyage/sentences/x.parquet"
    assert files["links"] == "polygon_document_links/x.parquet"
    assert len(files) == 6


def test_permalink():
    out = pl.select(wiki.permalink(pl.lit("Q2:wikivoyage:en:5:42"))).item()
    assert out == "https://en.wikivoyage.org/w/index.php?oldid=42"
