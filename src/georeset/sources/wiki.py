"""``osm-polygon-wikidata-and-wikipedia-landuse``: Wikipedia and Wikivoyage articles."""

from pathlib import Path

import polars as pl

from georeset.normalize.osm import geojson_to_wkb, polygon_id, primary_tag
from georeset.sources.base import SENTENCE_SCHEMA, Source, Staged, finish_sentences

PROJECTS = ("wikipedia", "wikivoyage")
# Rows made only of [...] segments: section edit links ("[ edit | edit source ]") and citation crumbs.
BRACKETS_ONLY = r"^\s*(\[[^\[\]]{1,80}\]\s*)+$"
SENTENCE_COLUMNS = ["sentence_id", "document_id", "section_index", "sentence_index", "text"]
LABEL_COLUMNS = ["sentence_id", "decision", "failure_reason", "language"]
POLYGON_COLUMNS = ["osm_type", "osm_id", "name", "tags", "lat", "lon", "area_m2", "geometry"]
LINK_COLUMNS = ["osm_type", "osm_id", "document_id"]


def files(stem: str) -> dict[str, str]:
    paths = {
        "polygons": f"polygons/{stem}.parquet",
        "links": f"polygon_document_links/{stem}.parquet",
    }
    for project in PROJECTS:
        paths[f"{project}_sentences"] = f"{project}/sentences/{stem}.parquet"
        paths[f"{project}_labels"] = f"labels/{project}/sentences/{stem}.parquet"
    return paths


def permalink(document_id: pl.Expr) -> pl.Expr:
    """Revision permalink from ``Q42:wikipedia:en:<page>:<revision>`` document ids."""
    parts = document_id.str.split(":")
    return pl.format(
        "https://{}.{}.org/w/index.php?oldid={}",
        parts.list.get(2),
        parts.list.get(1),
        parts.list.get(4),
    )


def _project_sentences(paths: dict[str, Path], project: str) -> pl.DataFrame:
    """Sentences of one project, or nothing when the region has no such file."""
    if f"{project}_sentences" not in paths:
        return pl.DataFrame(schema=SENTENCE_SCHEMA)
    sentences = pl.read_parquet(paths[f"{project}_sentences"], columns=SENTENCE_COLUMNS)  # pragma: no mutate
    labels = pl.read_parquet(paths[f"{project}_labels"], columns=LABEL_COLUMNS)  # pragma: no mutate
    return (
        sentences.filter(~pl.col("text").str.contains(BRACKETS_ONLY))
        .join(labels, on="sentence_id")
        .sort("section_index", "sentence_index")
        .select(
            source=pl.lit(project),
            doc_key="document_id",
            position=pl.int_range(pl.len(), dtype=pl.Int32).over("document_id"),
            text="text",
            label="decision",
            failure_reason="failure_reason",
            language="language",
            url=permalink(pl.col("document_id")),
        )
    )


def transform(paths: dict[str, Path]) -> Staged:
    polygons = pl.read_parquet(paths["polygons"], columns=POLYGON_COLUMNS)  # pragma: no mutate (speed)
    links = pl.read_parquet(paths["links"], columns=LINK_COLUMNS)  # pragma: no mutate (speed)
    sentences = pl.concat(_project_sentences(paths, project) for project in PROJECTS)
    return Staged(
        polygons=polygons.select(
            polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id")),
            name="name",
            osm_tag=primary_tag(pl.col("tags")),
            lat="lat",
            lon="lon",
            area_m2="area_m2",
            geometry=geojson_to_wkb(polygons["geometry"]),
        ),
        links=links.select(
            doc_key="document_id",
            polygon_id=polygon_id(pl.col("osm_type"), pl.col("osm_id")),
        ).unique(),
        sentences=finish_sentences(sentences, "wikimedia"),
    )


SOURCE = Source("wiki", "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse", files, transform)
