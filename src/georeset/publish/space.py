"""Data of the explorer Space: per-H3-cell numbers of the relevant sentences at every resolution it shows.

Resolution 5 is the data layer; each coarser cell holds the mean of its resolution-5 children (each child
counts once), as on the card. ``build`` writes, into ``out``:

* ``r2.parquet``, ``r3.parquet``, ``r4.parquet``: every cell of those resolutions;
* ``r5/<res-1 cell>.parquet`` and ``samples/<res-1 cell>.parquet``: resolution-5 cells, and up to
  ``SAMPLES`` relevant sentences per source and one scene per cell, split by resolution-1 parent and
  loaded as the map zooms in;
* ``manifest.json``: the dataset revision they come from, the layers' ranges, the PCA axes' names.
"""

import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from georeset.publish import geo, stats, themes

RESOLUTIONS = (2, 3, 4)  # coarser than the data layer, as the map zooms out
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
PCA = {"landuse": "pca_landuse", "english": "pca_english", "scenes": "pca_scenes"}


def _cell_ids(frame: pl.DataFrame) -> pl.Series:
    return pl.Series("cell", geo.h3_cells(frame["lon"].to_numpy(), frame["lat"].to_numpy()), pl.UInt64)


def sentence_cells(data: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per resolution-5 cell: relevant sentences, in total and per source; and (cell, language) counts."""
    sums = [pl.len().alias("relevant")]
    sums += [(pl.col("source") == source).sum().alias(name) for name, source in SOURCES.items()]
    relevant = data.filter(pl.col("landuse"))
    places = relevant.group_by("lat", "lon", "language").agg(sums).collect(engine="streaming")
    located = places.unique(["lat", "lon"]).select("lat", "lon")
    located = located.with_columns(_cell_ids(located))
    placed = places.join(located, on=["lat", "lon"]).drop("lat", "lon")
    languages = placed.group_by("cell", "language").agg(n=pl.col("relevant").sum())
    counts = placed.drop("language").group_by("cell").agg(pl.all().sum())
    return counts.with_columns(pl.col(c).cast(pl.Int64) for c in counts.columns if c != "cell"), languages


def photo_cells(photos: pl.LazyFrame) -> pl.DataFrame:
    """Scene photos per resolution-5 cell."""
    points = stats.points(photos)
    counts = pl.DataFrame({"cell": _cell_ids(points), "photos": points["n"]})
    return counts.group_by("cell").agg(pl.col("photos").sum())


def pca_cells(work: Path) -> dict[str, tuple[pl.DataFrame, np.ndarray]]:
    """Each computed corpus' resolution-5 cell scores (``cell``, ``n``, ``pc1``..) and colour limits."""
    out = {}
    for corpus in themes.CORPORA.values():
        folder = work / "themes" / corpus.name
        if (folder / "cells.parquet").is_file():
            display = pl.read_parquet(folder / "hexagons.parquet").select(themes.PCS).to_numpy()
            out[corpus.name] = (pl.read_parquet(folder / "cells.parquet"), stats.percentile_limits(display))
    return out


def _rgb(scores: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.round(stats.unit_scale(scores, limits) * 255).astype(np.uint8)


def _colours(name: str, scores: pl.DataFrame, limits: np.ndarray) -> pl.DataFrame:
    """``cell`` and its colour packed as ``r << 16 | g << 8 | b``, for cells with ``PCA_MIN`` texts."""
    kept = scores.filter(pl.col("n") >= PCA_MIN)
    rgb = _rgb(kept.select(themes.PCS).to_numpy(), limits).astype(np.int32)
    packed = rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2]
    return pl.DataFrame({"cell": kept["cell"], name: pl.Series(packed, dtype=pl.Int32)})


def rollup(cells: pl.DataFrame, languages: pl.DataFrame, pca: dict, resolution: int) -> pl.DataFrame:
    """Cells of ``resolution``: summed counts and each count's mean over the children holding it.

    Also the most frequent language and each PCA's colour from its children's mean scores.
    """
    parent = pl.Series("parent", geo.h3_parents(cells["cell"].to_numpy(), resolution), pl.UInt64)
    children = cells.with_columns(parent)
    # Means over the children holding each count: empty (null), not NaN, where none does.
    means = [pl.col(c).filter(pl.col(c) > 0).mean().cast(pl.Float32).alias(f"m_{c}") for c in COUNTS]
    out = children.group_by("parent").agg(
        pl.len().alias("children"), *[pl.col(c).sum() for c in COUNTS], *means
    )
    parents = pl.Series(geo.h3_parents(languages["cell"].to_numpy(), resolution), dtype=pl.UInt64)
    top = stats.dominant_language(
        languages.with_columns(hexagon=parents).group_by("hexagon", "language").agg(pl.col("n").sum())
    )
    out = out.join(top.rename({"hexagon": "parent"}), on="parent", how="left")
    for name, (scores, limits) in pca.items():
        child = scores.with_columns(
            parent=pl.Series(geo.h3_parents(scores["cell"].to_numpy(), resolution), dtype=pl.UInt64)
        )
        mean = child.group_by("parent").agg(pl.col("n").sum(), *[pl.col(pc).mean() for pc in themes.PCS])
        out = out.join(
            _colours(PCA[name], mean.rename({"parent": "cell"}), limits).rename({"cell": "parent"}),
            on="parent",
            how="left",
        )
    return out.rename({"parent": "cell"})


def sentence_samples(data: pl.LazyFrame) -> pl.DataFrame:
    """Up to ``SAMPLES`` whole relevant sentences per resolution-5 cell and source, by a hash of ``id``."""
    whole = pl.col("text").str.len_chars() >= SAMPLE_MIN_CHARS
    relevant = data.filter(pl.col("landuse") & whole).select(
        "id", "text", "source", "url", "language", "lat", "lon"
    )
    rows = relevant.collect(engine="streaming")
    located = rows.unique(["lat", "lon"]).select("lat", "lon")
    rows = rows.join(located.with_columns(_cell_ids(located)), on=["lat", "lon"])
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


def _ranges(tables: dict[int, pl.DataFrame]) -> dict:
    """Per resolution, the 99th percentile of each mean count: the top of its colour scale."""
    return {
        str(res): {c: float(table[f"m_{c}"].quantile(0.99) or 1) for c in COUNTS}
        for res, table in tables.items()
    }


def _axes(work: Path, names: list[str]) -> dict:
    out = {}
    for name in names:
        corpus = themes.CORPORA[name]
        summary = json.loads((work / "themes" / name / "summary.json").read_text())
        out[name] = {
            "what": corpus.what,
            "explained": summary["explained"],
            "axes": [list(corpus.axes[i]) for i in range(1, themes.COMPONENTS + 1)],
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
) -> Path:
    """Every file of the explorer's data into ``out``; ``revision`` is the dataset commit they come from."""
    counts, languages = sentence_cells(data)
    cells = counts.join(photo_cells(photos), on="cell", how="full", coalesce=True).fill_null(0)
    pca = pca_cells(work)
    tables = {res: rollup(cells, languages, pca, res) for res in (*RESOLUTIONS, geo.CELL_RESOLUTION)}
    out.mkdir(parents=True, exist_ok=True)
    for res in RESOLUTIONS:
        _write(tables[res], out / f"r{res}.parquet")
    chunks = _chunked(tables[geo.CELL_RESOLUTION], out / "r5")
    samples = sentence_samples(data).join(scene_samples(photos, scenes), on="cell", how="full", coalesce=True)
    _chunked(samples, out / "samples")
    top = languages.group_by("language").agg(pl.col("n").sum()).sort("n", descending=True).head(12)
    manifest = {
        "dataset": repo,
        "revision": revision,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "resolutions": [*RESOLUTIONS, geo.CELL_RESOLUTION],
        "chunk_resolution": CHUNK_RESOLUTION,
        "chunks": chunks,
        "ranges": _ranges(tables),
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
