"""Static figures for the dataset card: sentence-density world map and top languages."""

import io
import urllib.request
import zipfile
from pathlib import Path

import h3
import langcodes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapefile
import shapely
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
LAND = "#e9e8e4"
BAR = "#2a78d6"
# One-hue sequential ramp (light -> dark), one step per decade of sentences.
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
EDGES = [10**k for k in range(len(RAMP) + 1)]
LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
_PROJECT = Transformer.from_crs("EPSG:4326", "+proj=eqearth", always_xy=True)


def hex_counts(points: pl.DataFrame, resolution: int) -> pl.DataFrame:
    """Sum ``n`` per H3 cell of each (lat, lon) point."""
    cells = [
        h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(points["lat"], points["lon"], strict=True)
    ]
    return points.with_columns(cell=pl.Series(cells)).group_by("cell").agg(pl.col("n").sum())


def cell_ring(cell: str) -> np.ndarray | None:
    """Projected boundary of a cell; ``None`` for the few crossing the antimeridian."""
    lat, lon = np.array(h3.cell_to_boundary(cell)).T
    if lon.max() - lon.min() > 180:
        return None
    return np.column_stack(_PROJECT.transform(lon, lat))


def bin_index(n: pl.Series) -> pl.Series:
    """Decade bin of each count, capped at the last ramp step."""
    return n.log10().floor().clip(0, len(RAMP) - 1).cast(pl.Int64)


def load_land(cache: Path) -> list[shapely.Geometry]:
    """Natural Earth 1:110m land polygons, downloaded once into ``cache``."""
    path = cache / "ne_110m_land.zip"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LAND_URL) as response:
            path.write_bytes(response.read())
    with zipfile.ZipFile(path) as archive:
        reader = shapefile.Reader(
            shp=io.BytesIO(archive.read("ne_110m_land.shp")),
            shx=io.BytesIO(archive.read("ne_110m_land.shx")),
            dbf=io.BytesIO(archive.read("ne_110m_land.dbf")),
        )
        return [shapely.geometry.shape(s.__geo_interface__) for s in reader.shapes() if s is not None]


def _land_rings(land: list[shapely.Geometry]) -> list[np.ndarray]:
    polygons = [p for geom in land for p in getattr(geom, "geoms", [geom])]
    return [np.column_stack(_PROJECT.transform(*p.exterior.xy)) for p in polygons]


def _figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def world_map(counts: pl.DataFrame, land: list[shapely.Geometry], path: Path) -> Path:
    """Equal Earth map of sentences per H3 cell, log-binned on one blue ramp."""
    fig, ax = _figure(12, 6)
    ax.add_collection(PolyCollection(_land_rings(land), facecolors=LAND, edgecolors="none"))
    bins = bin_index(counts["n"]).to_list()
    drawn = [
        (ring, RAMP[b])
        for c, b in zip(counts["cell"], bins, strict=True)
        if (ring := cell_ring(c)) is not None
    ]
    rings, colors = zip(*drawn, strict=True) if drawn else ((), ())
    ax.add_collection(PolyCollection(list(rings), facecolors=list(colors), edgecolors="none"))
    ax.set_xlim(-17_000_000, 17_000_000)
    ax.set_ylim(-7_500_000, 8_700_000)
    ax.set_aspect("equal")
    ax.axis("off")
    norm = BoundaryNorm(EDGES, len(RAMP))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(RAMP)),
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        aspect=40,
    )
    bar.set_ticks(EDGES, labels=[f"{e:,}" for e in EDGES])
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    bar.set_label("Sentences per H3 cell (resolution 3)", color=TEXT_SECONDARY, fontsize=9)
    return _save(fig, path)


def language_label(code: str) -> str:
    """``French · fra_Latn``; the bare code when the language has no known name."""
    name = langcodes.Language.get(code.split("_")[0]).display_name()
    return code if name.startswith("Unknown") else f"{name} · {code}"


def language_bars(top: pl.DataFrame, path: Path) -> Path:
    """Horizontal bars of sentences per language, largest on top."""
    fig, ax = _figure(8, 0.32 * top.height + 0.6)
    labels = [language_label(c) for c in top["language"]][::-1]
    values = top["sentences"].to_list()[::-1]
    ax.barh(labels, values, color=BAR, height=0.7)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value:,}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9, length=0)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)
    return _save(fig, path)
