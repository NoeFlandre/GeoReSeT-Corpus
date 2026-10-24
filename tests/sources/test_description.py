import polars as pl
import shapely
from conftest import labels_columns

from georeset.sources import description


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
        labels_columns(
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


def test_description_files():
    assert description.files("x") == {
        "polygons": "data/x.parquet",
        "texts": "language-v1/data/x.parquet",
        "labels": "labels/language-v1/data/x.parquet",
    }
