"""Aggregates shown in the dataset card, computed from the published shards."""

import polars as pl

_WORDS = pl.col("text").str.count_matches(r"\S+").sum().alias("words")
_POLYGONS = pl.col("polygon_ids").list.explode(keep_nulls=False).n_unique().cast(pl.Int64)


def _summary(data: pl.LazyFrame, by: list[str]) -> pl.LazyFrame:
    return data.group_by(by).agg(
        sentences=pl.len().cast(pl.Int64),
        words=_WORDS.cast(pl.Int64),
        polygons=_POLYGONS,
        languages=pl.col("language").n_unique().cast(pl.Int64),
        regions=pl.col("region").n_unique().cast(pl.Int64),
        landuse=pl.col("landuse").mean(),
    )


def by_source(data: pl.LazyFrame) -> pl.DataFrame:
    """Per-source totals, largest first, plus a ``total`` row."""
    per_source = _summary(data, ["source"]).sort("sentences", "source", descending=[True, False])
    total = _summary(data.with_columns(source=pl.lit("total")), ["source"])
    return pl.concat([per_source, total]).collect()


def top_languages(data: pl.LazyFrame, n: int) -> pl.DataFrame:
    return (
        data.group_by("language")
        .agg(sentences=pl.len().cast(pl.Int64))
        .sort("sentences", "language", descending=[True, False])
        .head(n)
        .collect()
    )


def by_continent(data: pl.LazyFrame) -> pl.DataFrame:
    return (
        data.group_by("continent")
        .agg(
            sentences=pl.len().cast(pl.Int64),
            polygons=_POLYGONS,
        )
        .sort("sentences", "continent", descending=[True, False])
        .collect()
    )


def points(data: pl.LazyFrame) -> pl.DataFrame:
    """Sentence count per primary-polygon location, the input of the map."""
    return data.group_by("lat", "lon").agg(n=pl.len().cast(pl.Int64)).collect()


def unlabelled(data: pl.LazyFrame) -> pl.DataFrame:
    """Unlabelled rows per source and status, plus a ``total`` row."""
    wide = (
        data.group_by("source")
        .agg(
            failed=(pl.col("status") == "failed").sum().cast(pl.Int64),
            unsplit=(pl.col("status") == "unsplit").sum().cast(pl.Int64),
        )
        .sort("source")
        .collect()
    )
    total = wide.select(pl.lit("total").alias("source"), pl.col("failed", "unsplit").sum())
    return pl.concat([wide, total])
