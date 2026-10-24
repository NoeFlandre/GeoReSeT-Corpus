import polars as pl
import pytest

from georeset.assemble import (
    CLEAN_COLUMNS,
    POLYGON_COLUMNS,
    UNLABELLED_COLUMNS,
    canonical_docs,
    doc_meta,
    pick_polygons,
    region_priority,
    row_id,
    to_rows,
)


def test_region_priority_prefers_small_extracts_then_name():
    counts = pl.LazyFrame(
        {
            "stem": ["italy-latest", "centro-latest", "b-latest", "a-latest"],
            "n": [9, 2, 5, 5],
        }
    )
    out = region_priority(counts).collect().sort("priority")
    assert out["stem"].to_list() == [
        "centro-latest",
        "a-latest",
        "b-latest",
        "italy-latest",
    ]
    assert out["priority"].to_list() == [0, 1, 2, 3]


def _candidates():
    return pl.LazyFrame(
        {
            "polygon_id": ["way/1", "way/1", "way/1", "way/2"],
            "stem": ["italy-latest", "centro-latest", "centro-latest", "italy-latest"],
            "source": ["wiki", "description", "website", "website"],
            "name": ["from-italy", "from-desc", "from-web", "two"],
            "osm_tag": [None] * 4,
            "lat": [0.0] * 4,
            "lon": [0.0] * 4,
            "area_m2": [0.0] * 4,
            "geometry": [b""] * 4,
        }
    )


def test_pick_polygons_best_region_then_best_source():
    priority = pl.LazyFrame({"stem": ["centro-latest", "italy-latest"], "priority": [0, 1]})
    out = pick_polygons(_candidates(), priority).collect().sort("polygon_id")
    assert out["name"].to_list() == ["from-web", "two"]
    assert out["region"].to_list() == ["centro", "italy"]
    assert out["continent"].to_list() == ["europe", "europe"]
    assert out["osm_type"].to_list() == ["way", "way"]
    assert out["osm_id"].to_list() == [1, 2]
    assert out.columns == POLYGON_COLUMNS


def test_doc_meta_uses_first_sorted_polygon():
    links = pl.LazyFrame(
        {
            "doc_key": ["d", "d", "d", "e"],
            "polygon_id": ["way/9", "node/1", "way/9", "way/404"],
        }
    )
    polygons = pl.LazyFrame(
        {
            "polygon_id": ["node/1", "way/9"],
            "region": ["andorra", "italy"],
            "continent": ["europe", "europe"],
            "lat": [1.0, 2.0],
            "lon": [3.0, 4.0],
            "name": ["n", "w"],
            "osm_tag": ["a=b", "c=d"],
            "area_m2": [5.0, 6.0],
        }
    )
    out = doc_meta(links, polygons).collect()
    assert out.rows(named=True) == [
        {
            "doc_key": "d",
            "polygon_ids": ["node/1", "way/9"],
            "region": "andorra",
            "continent": "europe",
            "lat": 1.0,
            "lon": 3.0,
            "name": "n",
            "osm_tag": "a=b",
            "area_m2": 5.0,
        }
    ]


def test_canonical_docs_keeps_highest_priority_copy():
    presence = pl.LazyFrame(
        {
            "doc_key": ["d", "d", "e"],
            "stem": ["italy-latest", "centro-latest", "italy-latest"],
        }
    )
    priority = pl.LazyFrame({"stem": ["centro-latest", "italy-latest"], "priority": [0, 1]})
    out = canonical_docs(presence, priority).collect().sort("doc_key")
    assert out.rows() == [("d", "centro-latest"), ("e", "italy-latest")]


def test_row_id_is_stable_and_distinct():
    df = pl.DataFrame({"source": ["a", "a", "b"], "doc_key": ["d", "d", "d"], "position": [0, 1, 0]})
    ids = df.select(row_id()).to_series().to_list()
    assert len(set(ids)) == 3
    assert all(len(i) == 32 for i in ids)
    assert ids == df.select(row_id()).to_series().to_list()


def _sentences():
    return pl.DataFrame(
        {
            "source": ["wikipedia", "wikipedia", "website", "website"],
            "doc_key": ["d", "d", "u", "u"],
            "position": [0, 1, 0, 1],
            "text": ["yes text", "no text", "failed text", "whole page"],
            "label": ["yes", "no", "failed", "skipped_unsplit"],
            "failure_reason": [None, None, "truncated", "unsplit_upstream"],
            "language": ["fra_Latn"] * 4,
            "url": ["w", "w", "x", "x"],
        },
        schema_overrides={"position": pl.Int32},
    )


def _meta():
    return pl.DataFrame(
        {
            "doc_key": ["d", "u"],
            "polygon_ids": [["way/1"], ["way/2", "way/3"]],
            "region": ["italy", "andorra"],
            "continent": ["europe", "europe"],
            "lat": [1.0, 2.0],
            "lon": [3.0, 4.0],
            "name": ["a", "b"],
            "osm_tag": ["x=y", None],
            "area_m2": [1.0, 2.0],
        }
    )


def test_to_rows_splits_clean_and_unlabelled():
    clean, unlabelled = to_rows(_sentences(), _meta())
    assert clean.columns == [*CLEAN_COLUMNS, "doc_key", "position"]
    assert unlabelled.columns == [*UNLABELLED_COLUMNS, "doc_key", "position"]
    assert clean["landuse"].to_list() == [True, False]
    assert clean["license"].to_list() == ["CC-BY-SA-4.0"] * 2
    assert unlabelled["status"].to_list() == ["failed", "unsplit"]
    assert unlabelled["failure_reason"].to_list() == ["truncated", None]
    assert unlabelled["license"].to_list() == ["other"] * 2
    assert unlabelled["polygon_ids"].to_list() == [["way/2", "way/3"]] * 2


def test_to_rows_drops_sentences_without_meta():
    clean, unlabelled = to_rows(_sentences(), _meta().filter(pl.col("doc_key") == "u"))
    assert clean.height == 0
    assert unlabelled.height == 2


@pytest.mark.parametrize(
    ("source", "license"),
    [
        ("osm_description", "ODbL-1.0"),
        ("wikivoyage", "CC-BY-SA-4.0"),
        ("website", "other"),
    ],
)
def test_licenses(source, license):
    sentences = _sentences().with_columns(source=pl.lit(source))
    clean, _ = to_rows(sentences, _meta())
    assert set(clean["license"]) == {license}
