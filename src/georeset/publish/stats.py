"""Aggregates shown in the dataset card, computed from the published shards.

Everything runs on polars' streaming engine: the corpus is far larger than memory.
Each query scans the data once; two branches over one scan would make polars cache it.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from georeset.publish import geo

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
    return totals.row(0, named=True)  # pragma: no mutate (one row)


def scene_labels(scenes: pl.LazyFrame, n: int) -> pl.DataFrame:
    """The ``n`` land-use / land-cover labels covering most photos."""
    counts = _collect(
        scenes.group_by("lulc_label").agg(
            scenes=pl.len().cast(pl.Int64), photos=pl.col("n_photos").sum().cast(pl.Int64)
        )
    )
    ranked = counts.sort("photos", "lulc_label", descending=[True, False])  # pragma: no mutate (tie key)
    return ranked.head(n)


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


def landuse_sentences(data: pl.LazyFrame) -> pl.LazyFrame:
    """Sentences labelled relevant, one per distinct text (the smallest ``id`` keeps it)."""
    relevant = data.filter(pl.col("landuse")).select("id", "text", "language", "lat", "lon")
    return relevant.sort("id").unique("text", keep="first", maintain_order=True)


def scene_sentences(scenes: pl.LazyFrame) -> pl.LazyFrame:
    """Scene descriptions shaped like ``landuse_sentences`` (``id`` is ``scene_id``; all English)."""
    rows = scenes.select(
        pl.col("scene_id").alias("id"), "text", pl.lit("eng_Latn").alias("language"), "lat", "lon"
    )
    return rows.sort("id").unique("text", keep="first", maintain_order=True)


def photo_cells(photos: pl.DataFrame) -> pl.DataFrame:
    """Each (``id``, H3 ``cell``) pair where scene ``id`` has ``photos`` (``scene_id``, ``lat``, ``lon``)."""
    cells = geo.h3_cells(photos["lon"].to_numpy(), photos["lat"].to_numpy())
    return pl.DataFrame({"id": photos["scene_id"], "cell": cells}).unique().sort("id", "cell")


def parent_sums(
    cells: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The display layer: each parent of ``cells`` (``geo.DISPLAY_RESOLUTION``), the sum of its children's
    ``values`` rows, its number of children, and each child's index among the parents.
    """
    parents, inverse = np.unique(geo.h3_parents(cells), return_inverse=True)
    _, sums, sizes = group_sums(inverse, values)
    return parents, sums, sizes, inverse


def group_sums(groups: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct ``groups`` (negative ones dropped), each one's sum of ``values`` rows, and its size."""
    keep = groups >= 0
    order = np.argsort(groups[keep], kind="stable")
    present, starts, sizes = np.unique(groups[keep][order], return_index=True, return_counts=True)
    if not present.size:
        return present, np.zeros((0, values.shape[1])), sizes
    return present, np.add.reduceat(values[keep][order].astype(np.float64), starts), sizes


@dataclass(frozen=True)
class Pca:
    mean: np.ndarray  # (dims,)
    components: np.ndarray  # (k, dims), unit rows
    explained: np.ndarray  # (k,) share of the total variance

    def scores(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components.T


def pca(x: np.ndarray, k: int) -> Pca:
    """Principal components of the rows of ``x``; each component's largest loading is positive."""
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    signs = np.sign(vt[np.arange(vt.shape[0]), np.abs(vt).argmax(axis=1)])
    variance = singular**2
    return Pca(mean, (vt * signs[:, None])[:k], (variance / variance.sum())[:k])


CLIP = (2, 98)


def percentile_limits(scores: np.ndarray) -> np.ndarray:
    """Each column's ``CLIP`` percentiles: rows low, high."""
    return np.percentile(scores, CLIP, axis=0)


def unit_scale(scores: np.ndarray, limits: np.ndarray | None = None) -> np.ndarray:
    """Each column scaled to [0, 1] between ``limits`` (default: its own), clipped outside."""
    lo, hi = percentile_limits(scores) if limits is None else limits
    return np.clip((scores - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)


def symmetric_limit(scores: np.ndarray) -> float:
    """Half-width of a two-sided scale centred on 0, covering ``CLIP`` percentiles."""
    return float(np.abs(np.percentile(scores, CLIP)).max())


def dominant_language(counts: pl.DataFrame) -> pl.DataFrame:
    """Most frequent ``language`` per ``hexagon`` from (hexagon, language, n) counts; ties by name."""
    ranked = counts.sort(["hexagon", "n", "language"], descending=[False, True, False])
    return ranked.group_by("hexagon", maintain_order=True).first().select("hexagon", "language")


def r_squared(scores: np.ndarray, groups: list[str]) -> np.ndarray:
    """Share of each column's variance explained by group means (one-way ANOVA R²)."""
    frame = pl.DataFrame({"group": groups}).with_columns(
        [pl.Series(f"s{i}", scores[:, i]) for i in range(scores.shape[1])]
    )
    columns = [f"s{i}" for i in range(scores.shape[1])]
    fitted = frame.select(pl.col(columns).mean().over("group")).to_numpy()
    total = ((scores - scores.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ((scores - fitted) ** 2).sum(axis=0) / total


def extremes(scored: pl.DataFrame, column: str, n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The ``n`` rows with the highest, then the lowest ``column``."""
    ranked = scored.sort(column, "id", descending=[True, False])
    return ranked.head(n), ranked.reverse().head(n)


def language_mix(rows: pl.DataFrame) -> pl.DataFrame:
    """Share of each language among ``rows``, largest first."""
    counts = rows.group_by("language").agg(share=pl.len() / rows.height)
    return counts.sort("share", "language", descending=[True, False])
