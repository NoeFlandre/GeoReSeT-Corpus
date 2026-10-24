"""``osm-polygon-description-tag-landuse``: OSM ``description*`` tags of polygons."""

from pathlib import Path

import polars as pl

from georeset.normalize.osm import polygon_id, primary_tag, tags_list_to_json
from georeset.sources.base import Source, Staged, attach_text, finish_sentences, osm_url

BBOX = ["bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y"]


def files(stem: str) -> dict[str, str]:
    return {
        "polygons": f"data/{stem}.parquet",
        "texts": f"language-v1/data/{stem}.parquet",
        "labels": f"labels/language-v1/data/{stem}.parquet",
    }


def _polygons(path: Path) -> pl.DataFrame:
    return pl.read_parquet(
        path,
        columns=["osm_type", "osm_id", "name", "tags", "area_m2", "geometry", *BBOX],
    ).select(
        polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id")),
        name="name",
        osm_tag=primary_tag(tags_list_to_json(pl.col("tags"))),
        lat=(pl.col("bbox_min_y") + pl.col("bbox_max_y")) / 2,
        lon=(pl.col("bbox_min_x") + pl.col("bbox_max_x")) / 2,
        area_m2="area_m2",
        geometry="geometry",
    )


def _sentences(texts: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    keys = ["description_identity", "tag_key"]
    labelled = labels.join(texts.select(*keys, "polygon_id"), on=keys)
    docs = texts.select(*keys, "sentences", full_text="original_text")
    return attach_text(labelled, docs, keys).select(
        source=pl.lit("osm_description"),
        doc_key="description_identity",
        position="sentence_index",
        text="text",
        label="label",
        failure_reason="failure_reason",
        language="language",
        url=osm_url(pl.col("polygon_id")),
    )


def transform(paths: dict[str, Path]) -> Staged:
    texts = pl.read_parquet(
        paths["texts"],
        columns=[
            "description_identity",
            "osm_type",
            "osm_id",
            "tag_key",
            "original_text",
            "sentences",
        ],
    ).with_columns(polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id")))
    labels = pl.read_parquet(
        paths["labels"],
        columns=[
            "description_identity",
            "tag_key",
            "sentence_index",
            "decision",
            "failure_reason",
            "language",
        ],
    ).rename({"decision": "label"})
    return Staged(
        polygons=_polygons(paths["polygons"]),
        links=texts.select(doc_key="description_identity", polygon_id="polygon_id").unique(),
        sentences=finish_sentences(_sentences(texts, labels), "iso639_3"),
    )


SOURCE = Source("description", "NoeFlandre/osm-polygon-description-tag-landuse", files, transform)
