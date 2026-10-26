"""Aggregates shown in the dataset card, computed from the published shards.

Everything runs on polars' streaming engine: the corpus is far larger than memory.
Each query scans the data once; two branches over one scan would make polars cache it.
"""

import polars as pl

_WORDS = pl.col("text").str.count_matches(r"\S+").sum().cast(pl.Int64).alias("words")


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def _distinct(pairs: pl.LazyFrame, by: str, name: str) -> pl.DataFrame:
    """Distinct values per group from (group, value) pairs; streams, unlike ``n_unique``."""
    return _collect(pairs.unique().group_by(by).agg(pl.len().cast(pl.Int64).alias(name)))


def _polygons(data: pl.LazyFrame, by: str) -> pl.DataFrame:
    return _distinct(data.select(by, "polygon_ids").explode("polygon_ids"), by, "polygons")


COLUMNS = ["source", "sentences", "words", "polygons", "languages", "regions", "landuse"]


def _distincts(data: pl.LazyFrame, by: str) -> pl.DataFrame:
    """Distinct polygons, languages and regions per group."""
    out = _polygons(data, by)
    for column, name in (("language", "languages"), ("region", "regions")):
        out = out.join(_distinct(data.select(by, column), by, name), on=by)
    return out


def by_source(data: pl.LazyFrame) -> pl.DataFrame:
    """Per-source totals, largest first, plus a ``total`` row.

    Additive totals come from the per-source rows; grouping by a literal does not stream.
    """
    counts = _collect(
        data.group_by("source").agg(
            sentences=pl.len().cast(pl.Int64), words=_WORDS, landuse=pl.col("landuse").mean()
        )
    )
    per_source = counts.join(_distincts(data, "source"), on="source").select(COLUMNS)
    total = counts.select(
        source=pl.lit("total"),
        sentences=pl.col("sentences").sum(),
        words=pl.col("words").sum(),
        landuse=(pl.col("landuse") * pl.col("sentences")).sum() / pl.col("sentences").sum(),
    ).join(_distincts(data.with_columns(source=pl.lit("total")), "source"), on="source")
    return pl.concat(
        [per_source.sort("sentences", "source", descending=[True, False]), total.select(COLUMNS)]
    )


def top_languages(data: pl.LazyFrame, n: int) -> pl.DataFrame:
    counts = _collect(data.group_by("language").agg(sentences=pl.len().cast(pl.Int64)))
    return counts.sort("sentences", "language", descending=[True, False]).head(n)


def by_continent(data: pl.LazyFrame) -> pl.DataFrame:
    sentences = _collect(data.group_by("continent").agg(sentences=pl.len().cast(pl.Int64)))
    joined = sentences.join(_polygons(data, "continent"), on="continent")
    return joined.sort("sentences", "continent", descending=[True, False])


def points(data: pl.LazyFrame) -> pl.DataFrame:
    """Sentence count per primary-polygon location, the input of the map."""
    return _collect(data.group_by("lat", "lon").agg(n=pl.len().cast(pl.Int64)))


def unlabelled(data: pl.LazyFrame) -> pl.DataFrame:
    """Unlabelled rows per source and status, plus a ``total`` row."""
    wide = _collect(
        data.group_by("source").agg(
            failed=(pl.col("status") == "failed").sum().cast(pl.Int64),
            unsplit=(pl.col("status") == "unsplit").sum().cast(pl.Int64),
        )
    ).sort("source")
    total = wide.select(pl.lit("total").alias("source"), pl.col("failed", "unsplit").sum())
    return pl.concat([wide, total])


AREA_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def polygon_areas(polygons: pl.LazyFrame) -> pl.DataFrame:
    """Area quantiles of all polygons, in m² (no total: polygons overlap, e.g. a park in a city)."""
    area = pl.col("area_m2")
    quantiles = [area.quantile(q, "lower").alias(f"p{round(q * 100)}") for q in AREA_QUANTILES]
    return _collect(polygons.select(quantiles))


def scene_totals(scenes: pl.LazyFrame) -> dict[str, int]:
    """Descriptions, photos they cover (and per description, median), words and countries."""
    totals = _collect(
        scenes.select(
            scenes=pl.len().cast(pl.Int64),
            photos=pl.col("n_photos").sum().cast(pl.Int64),
            median_photos=pl.col("n_photos").median().round().cast(pl.Int64),
            words=_WORDS,
            countries=pl.col("countries").explode().n_unique().cast(pl.Int64),
        )
    )
    return totals.row(0, named=True)


def scene_labels(scenes: pl.LazyFrame, n: int) -> pl.DataFrame:
    """The ``n`` land-use / land-cover labels covering most photos."""
    counts = _collect(
        scenes.group_by("lulc_label").agg(
            scenes=pl.len().cast(pl.Int64), photos=pl.col("n_photos").sum().cast(pl.Int64)
        )
    )
    return counts.sort("photos", "lulc_label", descending=[True, False]).head(n)


def scene_platforms(photos: pl.LazyFrame) -> pl.DataFrame:
    """Photos per platform and the share that keeps a public URL."""
    counts = _collect(
        photos.group_by("platform").agg(
            photos=pl.len().cast(pl.Int64), linked=pl.col("url").is_not_null().mean()
        )
    )
    return counts.sort("photos", "platform", descending=[True, False])


SPREAD_KM = (10, 100, 1_000)


def scene_locality(scenes: pl.LazyFrame) -> pl.DataFrame:
    """Scenes whose photos lie, for 90% of them, within each distance of their median location."""
    within = [(pl.col("spread_km_p90") <= km).sum().cast(pl.Int64).alias(f"km{km}") for km in SPREAD_KM]
    return _collect(scenes.select(within))
