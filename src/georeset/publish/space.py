"""Data of the explorer Space: per-H3-cell numbers of the relevant sentences, all at ``geo.CELL_RESOLUTION``.

Every count and colour is that of a resolution-5 cell, as on the card; the PCA colours are the cell's own.

``build`` writes, into ``out``:

* ``r5/<res-1 cell>.parquet`` and ``samples/<res-1 cell>.parquet``: the cells, and up to ``SAMPLES``
  relevant sentences per source and one scene per cell, split by resolution-1 parent and loaded as the
  map is explored;
* ``manifest.json``: the dataset revision they come from, the layers' ranges, the PCA's variance.
"""

import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from georeset.publish import geo, stats, themes

CHUNK_RESOLUTION = 1  # resolution-5 cells and samples are split by parent at this resolution
SAMPLES = 3  # relevant sentences kept per resolution-5 cell and source
SAMPLE_CHARS = 300
SAMPLE_MIN_CHARS = 40  # shorter texts are mostly fragments (names, table cells)
SCENE_CHARS = 400
PCA_MIN = 5  # texts a cell needs for its PCA colour
# The explorer shows only the sentences labelled relevant (``landuse`` true), in total and per source.
SOURCES = {
    "website": "website",
    "description": "osm_description",
    "wikipedia": "wikipedia",
    "wikivoyage": "wikivoyage",
}
COUNTS = ("relevant", *SOURCES, "photos")
# The dataset folders the explorer is built from: texts, scenes, and the vectors of the PCA.
FOLDERS = ("data", "scenes", "scene_photos", "embeddings", "scene_embeddings")
PCA = {"landuse": "pca_landuse", "english": "pca_english", "scenes": "pca_scenes"}


def _cell_ids(frame: pl.DataFrame) -> pl.Series:
    return pl.Series("cell", geo.h3_cells(frame["lon"].to_numpy(), frame["lat"].to_numpy()), pl.UInt64)


def _with_cells(places: pl.DataFrame) -> pl.DataFrame:
    """``places`` (with ``lat`` and ``lon``) and each place's ``cell``, the same for rows at one spot."""
    located = places.unique(["lat", "lon"]).select("lat", "lon")
    located = located.with_columns(_cell_ids(located))
    return places.join(located, on=["lat", "lon"])


def sentence_cells(data: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per resolution-5 cell: relevant sentences, in total and per source; and (cell, language) counts."""
    sums = [pl.len().alias("relevant")]
    sums += [(pl.col("source") == source).sum().alias(name) for name, source in SOURCES.items()]
    relevant = data.filter(pl.col("landuse"))
    places = relevant.group_by("lat", "lon", "language").agg(sums).collect(engine="streaming")
    placed = _with_cells(places).drop("lat", "lon")
    languages = placed.group_by("cell", "language").agg(n=pl.col("relevant").sum())
    counts = placed.drop("language").group_by("cell").agg(pl.all().sum())
    return counts.with_columns(pl.col(c).cast(pl.Int64) for c in counts.columns if c != "cell"), languages


def pca_cells(work: Path) -> dict[str, tuple[pl.DataFrame, np.ndarray]]:
    """Each computed corpus' ``geo.CELL_RESOLUTION`` cell scores (``cell``, ``n``, ``pc1``..) and the colour
    limits of the card's maps (global: 2nd and 98th percentiles)."""
    out = {}
    for corpus in themes.CORPORA.values():
        folder = work / "themes" / corpus.name
        if (folder / "cells.parquet").is_file():
            display = pl.read_parquet(folder / "hexagons.parquet")
            out[corpus.name] = (
                pl.read_parquet(folder / "cells.parquet"),
                stats.percentile_limits(display.select(themes.PCS).to_numpy()),
            )
    return out


def _packed(rgb: np.ndarray) -> np.ndarray:
    """Colours in [0, 1] packed as ``r << 16 | g << 8 | b``."""
    rgb = np.round(rgb * 255).astype(np.int32)
    return rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2]


def _colours(name: str, scores: pl.DataFrame, limits: np.ndarray) -> pl.DataFrame:
    """For cells with ``PCA_MIN`` texts: their colour from the global axes (red, green, blue)."""
    kept = scores.filter(pl.col("n") >= PCA_MIN)
    rgb = stats.unit_scale(kept.select(themes.PCS).to_numpy(), limits)
    return pl.DataFrame({"cell": kept["cell"], name: pl.Series(_packed(rgb), dtype=pl.Int32)})


def cell_table(cells: pl.DataFrame, languages: pl.DataFrame, pca: dict) -> pl.DataFrame:
    """One row per cell: its counts, its most frequent language and each PCA's colour."""
    top = stats.dominant_language(
        languages.rename({"cell": "hexagon"}).group_by("hexagon", "language").agg(pl.col("n").sum())
    )
    out = cells.join(top.rename({"hexagon": "cell"}), on="cell", how="left")
    for name, (scores, limits) in pca.items():
        mean = scores.group_by("cell").agg(pl.col("n").sum(), *[pl.col(a).mean() for a in themes.PCS])
        out = out.join(_colours(PCA[name], mean, limits), on="cell", how="left")
    return out


def sentence_samples(data: pl.LazyFrame) -> pl.DataFrame:
    """Up to ``SAMPLES`` whole relevant sentences per resolution-5 cell and source, by a hash of ``id``."""
    whole = pl.col("text").str.len_chars() >= SAMPLE_MIN_CHARS
    relevant = data.filter(pl.col("landuse") & whole).select(
        "id", "text", "source", "url", "language", "lat", "lon"
    )
    rows = relevant.collect(engine="streaming")
    rows = _with_cells(rows)
    pick = pl.col("id").hash().rank("ordinal").over("cell", "source")
    return (
        rows.filter(pick <= SAMPLES)
        .with_columns(pl.col("text").str.slice(0, SAMPLE_CHARS))
        .select("cell", "text", "source", "url", "language")
        .sort("cell", "text")
    )


def scene_samples(photos: pl.LazyFrame, scenes: pl.LazyFrame) -> pl.DataFrame:
    """Per resolution-5 cell, the scene with the most photos there: its label and description."""
    points = photos.group_by("scene_id", "lat", "lon").agg(n=pl.len()).collect(engine="streaming")
    points = points.with_columns(_cell_ids(points))
    counts = points.group_by("cell", "scene_id").agg(pl.col("n").sum())
    top = counts.sort(["cell", "n", "scene_id"], descending=[False, True, False]).unique("cell", keep="first")
    text = scenes.select("scene_id", pl.col("text").str.slice(0, SCENE_CHARS), "lulc_label").collect()
    joined = top.join(text, on="scene_id").select(
        "cell", "scene_id", "lulc_label", pl.col("text").alias("scene")
    )
    return joined.sort("cell")


def _hex(cells: pl.Series) -> pl.Series:
    return pl.Series("cell", [format(c, "x") for c in cells.to_list()], pl.String)


def _write(frame: pl.DataFrame, path: Path) -> None:
    """``frame`` as the browser reads it: hex cell ids, 32-bit numbers, snappy compression."""
    hexed = frame.with_columns(_hex(frame["cell"]))
    narrow = [pl.col(pl.Int64, pl.UInt32, pl.UInt64).cast(pl.Int32), pl.col(pl.Float64).cast(pl.Float32)]
    hexed.with_columns(narrow).write_parquet(path, compression="snappy")


def _chunked(frame: pl.DataFrame, folder: Path) -> list[str]:
    """``frame`` split by each cell's resolution-``CHUNK_RESOLUTION`` parent, one file per parent."""
    folder.mkdir(parents=True, exist_ok=True)
    parents = _hex(pl.Series(geo.h3_parents(frame["cell"].to_numpy(), CHUNK_RESOLUTION)))
    names = []
    for part in frame.with_columns(chunk=parents).partition_by("chunk"):
        name = part["chunk"][0]
        _write(part.drop("chunk"), folder / f"{name}.parquet")
        names.append(name)
    return sorted(names)


def _ranges(table: pl.DataFrame) -> dict:
    """The 99th percentile of each count: the top of its colour scale."""
    return {str(geo.CELL_RESOLUTION): {c: float(table[c].quantile(0.99) or 1) for c in COUNTS}}


def _axes(work: Path, names: list[str]) -> dict:
    out = {}
    for name in names:
        corpus = themes.CORPORA[name]
        summary = json.loads((work / "themes" / name / "summary.json").read_text())
        out[name] = {
            "what": corpus.what,
            "explained": summary["explained"],
        }
    return out


def build(
    data: pl.LazyFrame,
    photos: pl.LazyFrame,
    scenes: pl.LazyFrame,
    work: Path,
    out: Path,
    repo: str,
    revision: str,
    commits: dict[str, str],
) -> Path:
    """Every file of the explorer's data into ``out``.

    ``revision`` is the dataset commit they come from; ``commits``, the last commit of each of its ``FOLDERS``
    then, which the app compares with the dataset's to tell when the data (not just the card) has changed.
    """
    counts, languages = sentence_cells(data)
    photo_counts = geo.hex_counts(stats.points(photos)).rename({"n": "photos"})  # scene photos per cell
    cells = counts.join(photo_counts, on="cell", how="full", coalesce=True).fill_null(0)
    pca = pca_cells(work)
    table = cell_table(cells, languages, pca)
    out.mkdir(parents=True, exist_ok=True)
    chunks = _chunked(table, out / "r5")
    samples = sentence_samples(data).join(scene_samples(photos, scenes), on="cell", how="full", coalesce=True)
    _chunked(samples, out / "samples")
    top = languages.group_by("language").agg(pl.col("n").sum()).sort("n", descending=True).head(12)
    manifest = {
        "dataset": repo,
        "revision": revision,
        "folders": commits,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "resolutions": [geo.CELL_RESOLUTION],
        "chunk_resolution": CHUNK_RESOLUTION,
        "chunks": chunks,
        "ranges": _ranges(table),
        "languages": top["language"].to_list(),
        "pca": _axes(work, list(pca)),
        "pca_min": PCA_MIN,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False))
    return out


APP = files("georeset.publish").joinpath("space_app")


def copy_app(out: Path) -> None:
    """The Space's page, script, style and card next to its ``data``."""
    out.mkdir(parents=True, exist_ok=True)
    for item in APP.iterdir():
        (out / item.name).write_bytes(item.read_bytes())
