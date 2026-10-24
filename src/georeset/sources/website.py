"""``osm-polygon-website-tag-landuse``: text of the websites tagged on polygons."""

from pathlib import Path

import polars as pl

from georeset.osm import geojson_to_wkb, polygon_id, primary_tag
from georeset.sources import Source, Staged, attach_text, finish_sentences

FIELDS = ("website", "contact_website")


def files(stem: str) -> dict[str, str]:
    return {
        "polygons": f"polygons/{stem}.parquet",
        "labels": f"labels/polygons/{stem}.parquet",
    }


POLYGON_COLUMNS = [
    "polygon_id",
    "osm_type",
    "osm_id",
    "name",
    "tags",
    "lat",
    "lon",
    "area_m2",
    "geometry",
]


def _docs(path: Path) -> pl.DataFrame:
    """One row per (polygon, field) with a URL: the documents of this source.

    Full page text is only read, streaming, for pages never split into sentences.
    """
    scan = pl.scan_parquet(path)
    frames = []
    for field in FIELDS:
        sentences = pl.col(f"{field}_sentences")
        unsplit = scan.filter(sentences.is_null()).select(
            raw_id="polygon_id", full_text=pl.col(f"{field}_text")
        )
        frames.append(
            scan.select(
                raw_id="polygon_id",
                polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id")),
                field=pl.lit(field),
                url=pl.col(field),
                sentences=sentences,
            ).join(unsplit, on="raw_id", how="left")
        )
    return pl.concat(frames).drop_nulls("url").collect(engine="streaming")


def _sentences(docs: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    canonical = docs.sort("polygon_id", "field").unique(
        "url", keep="first", maintain_order=True
    )
    labelled = labels.join(
        canonical.select("raw_id", "field", "url"),
        left_on=["polygon_id", "field"],
        right_on=["raw_id", "field"],
    )
    docs_text = canonical.select(
        polygon_id="raw_id", field="field", sentences="sentences", full_text="full_text"
    )
    return attach_text(labelled, docs_text, ["polygon_id", "field"]).select(
        source=pl.lit("website"),
        doc_key="url",
        position="sentence_index",
        text="text",
        label="label",
        failure_reason="failure_reason",
        language="language",
        url="url",
    )


def transform(paths: dict[str, Path]) -> Staged:
    polygons = pl.read_parquet(paths["polygons"], columns=POLYGON_COLUMNS).with_columns(
        polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id"))
    )
    labels = pl.read_parquet(
        paths["labels"],
        columns=[
            "polygon_id",
            "field",
            "sentence_index",
            "decision",
            "failure_reason",
            "language",
        ],
    ).rename({"decision": "label"})
    docs = _docs(paths["polygons"])
    return Staged(
        polygons=polygons.select(
            "polygon_id",
            "name",
            osm_tag=primary_tag(pl.col("tags")),
            lat="lat",
            lon="lon",
            area_m2="area_m2",
            geometry=geojson_to_wkb(polygons["geometry"]),
        ),
        links=docs.select(doc_key="url", polygon_id="polygon_id").unique(),
        sentences=finish_sentences(_sentences(docs, labels), "glotlid"),
    )


SOURCE = Source(
    "website", "NoeFlandre/osm-polygon-website-tag-landuse", files, transform
)
