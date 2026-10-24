"""Merge staged regions: one row per polygon, one copy per document, final row schema."""

import hashlib

import polars as pl

from georeset.normalize.regions import CONTINENT_OF, SUFFIX

SOURCE_RANK = {"wiki": 0, "website": 1, "description": 2}
LICENSES = {
    "website": "other",
    "osm_description": "ODbL-1.0",
    "wikipedia": "CC-BY-SA-4.0",
    "wikivoyage": "CC-BY-SA-4.0",
}
META_COLUMNS = [
    "polygon_ids",
    "region",
    "continent",
    "lat",
    "lon",
    "name",
    "osm_tag",
    "area_m2",
]
COMMON = ["id", "text", "language", "source", *META_COLUMNS, "url", "license"]
CLEAN_COLUMNS = [*COMMON[:2], "landuse", *COMMON[2:]]
UNLABELLED_COLUMNS = [*COMMON[:2], "status", "failure_reason", *COMMON[2:]]
CLEAN_LABELS = ("yes", "no")
POLYGON_COLUMNS = [
    "polygon_id",
    "osm_type",
    "osm_id",
    "name",
    "osm_tag",
    "region",
    "continent",
    "lat",
    "lon",
    "area_m2",
    "geometry",
]


def region_priority(counts: pl.LazyFrame) -> pl.LazyFrame:
    """Rank extracts so the smallest (most specific) wins overlaps: ``centro`` over ``italy``."""
    return counts.sort("n", "stem").select("stem", priority=pl.int_range(pl.len()))


def pick_polygons(candidates: pl.LazyFrame, priority: pl.LazyFrame) -> pl.LazyFrame:
    """One row per polygon: best region first, then the source with the best attributes."""
    rank = pl.col("source").replace_strict(SOURCE_RANK)
    region = pl.col("stem").str.strip_suffix(SUFFIX)
    osm = pl.col("polygon_id").str.split_exact("/", 1)  # pragma: no mutate (ids hold one "/")
    return (
        candidates.join(priority, on="stem")
        .sort("priority", rank)
        .unique(
            "polygon_id", keep="first"
        )  # pragma: no mutate (keep='first' is required; polars' 'any' only matches by luck)
        .with_columns(region=region)
        .with_columns(
            continent=pl.col("region").replace_strict(CONTINENT_OF),
            osm_type=osm.struct.field("field_0"),
            osm_id=osm.struct.field("field_1").cast(pl.Int64),
        )
        .select(POLYGON_COLUMNS)
    )


def doc_meta(links: pl.LazyFrame, polygons: pl.LazyFrame) -> pl.LazyFrame:
    """Per document: sorted ``polygon_ids`` and the attributes of the first one."""
    known = links.unique().join(polygons.select("polygon_id"), on="polygon_id")
    ids = known.group_by("doc_key").agg(pl.col("polygon_id").sort().alias("polygon_ids"))
    primary = ids.with_columns(polygon_id=pl.col("polygon_ids").list.first())
    return primary.join(polygons, on="polygon_id").select("doc_key", *META_COLUMNS)


def canonical_docs(presence: pl.LazyFrame, priority: pl.LazyFrame) -> pl.LazyFrame:
    """The one staged region each document's sentences are taken from."""
    return (
        presence.join(priority, on="stem")
        .sort("priority")
        .unique(
            "doc_key", keep="first"
        )  # pragma: no mutate (keep='first' is required; polars' 'any' only matches by luck)
        .select("doc_key", "stem")
    )


def _hash(keys: pl.Series) -> pl.Series:
    return pl.Series([hashlib.blake2b(k.encode(), digest_size=16).hexdigest() for k in keys])


def row_id() -> pl.Expr:
    """Stable 128-bit hex id of (source, document, position)."""
    key = pl.concat_str(pl.col("source"), pl.col("doc_key"), pl.col("position"), separator="\x1f")
    return key.map_batches(_hash, return_dtype=pl.String).alias("id")  # pragma: no mutate (lazy schema)


def to_rows(sentences: pl.DataFrame, meta: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Final (clean, unlabelled) rows; ``doc_key`` and ``position`` are kept for sorting."""
    rows = sentences.join(meta, on="doc_key").with_columns(
        row_id(),
        license=pl.col("source").replace_strict(LICENSES),
    )
    is_clean = pl.col("label").is_in(CLEAN_LABELS)
    clean = rows.filter(is_clean).with_columns(landuse=pl.col("label") == "yes")
    unlabelled = rows.filter(~is_clean).with_columns(
        status=pl.when(pl.col("label") == "failed").then(pl.lit("failed")).otherwise(pl.lit("unsplit")),
        failure_reason=pl.when(pl.col("label") == "failed").then("failure_reason"),
    )
    keys = ["doc_key", "position"]
    return clean.select(*CLEAN_COLUMNS, *keys), unlabelled.select(*UNLABELLED_COLUMNS, *keys)
