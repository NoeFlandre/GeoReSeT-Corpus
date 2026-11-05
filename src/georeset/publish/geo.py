"""Numbers behind the maps: H3 cells, the Equal Earth projection, log-spaced colour bins.

Every map has two layers: points are counted in H3 cells of ``CELL_RESOLUTION``, then each cell of
``DISPLAY_RESOLUTION`` (their parent) shows the mean of its child cells, drawn on the Equal Earth
projection, which keeps areas true.
"""

import itertools
import json
from functools import cache
from importlib.resources import files

import h3
import h3.api.basic_int as h3_int
import numpy as np
import polars as pl

CELL_RESOLUTION = 5  # where points are counted: cells of about 250 km²
DISPLAY_RESOLUTION = 4  # what the maps draw: cells of about 1,800 km², each the mean of its child cells
LATITUDES = (-60, 85)  # the maps' latitude range
# Natural Earth 1:110m land (public domain, v5.1.2).
LAND = files("georeset.publish").joinpath("assets/ne_110m_land.geojson")
# Equal Earth (Šavrič, Patterson and Jenny, 2018).
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_M = np.sqrt(3) / 2


def h3_cells(lon: np.ndarray, lat: np.ndarray, resolution: int = CELL_RESOLUTION) -> np.ndarray:
    """Each point's H3 cell, as 64-bit ids."""
    pairs = zip(np.asarray(lat, float), np.asarray(lon, float), strict=True)
    return np.fromiter((h3_int.latlng_to_cell(a, o, resolution) for a, o in pairs), np.uint64, len(lat))


def h3_parents(cells: np.ndarray, resolution: int = DISPLAY_RESOLUTION) -> np.ndarray:
    """Each cell's parent at ``resolution``."""
    return np.fromiter((h3_int.cell_to_parent(int(c), resolution) for c in cells), np.uint64, len(cells))


def h3_centres(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of each H3 cell's centre."""
    centres = np.array([h3_int.cell_to_latlng(int(c)) for c in cells], float).reshape(-1, 2)
    return centres[:, 1], centres[:, 0]


def hex_counts(points: pl.DataFrame, resolution: int = CELL_RESOLUTION) -> pl.DataFrame:
    """Sum ``n`` per H3 ``cell`` (64-bit id) of each (lat, lon) point."""
    cells = h3_cells(points["lon"].to_numpy(), points["lat"].to_numpy(), resolution)
    counts = pl.DataFrame({"cell": cells, "n": points["n"]}, schema={"cell": pl.UInt64, "n": pl.Int64})
    return counts.group_by("cell").agg(pl.col("n").sum())


def equal_earth(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal Earth coordinates of each (lon, lat) in degrees, on a unit sphere."""
    theta = np.arcsin(_M * np.sin(np.radians(lat)))
    t2 = theta**2
    t6 = t2**3
    x = np.radians(lon) * np.cos(theta) / (_M * (_A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)))
    return x, theta * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))


def outline(steps: int = 181) -> np.ndarray:
    """The projected edge of the world: the ±180° meridians, as one closed ring."""
    x, y = equal_earth(np.full(steps, 180.0), np.linspace(-90, 90, steps))
    return np.concatenate([np.column_stack([x, y]), np.column_stack([-x, y])[::-1]])


def cell_rings(cells: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Each cell's projected outline and the index of the cell each ring draws.

    A cell across ±180° gets a ring on each side; drawing clips them to the ``outline``.
    """
    rings, owners = [], []
    for i, cell in enumerate(cells):
        lat, lon = np.array(h3_int.cell_to_boundary(int(cell))).T
        if lon.max() - lon.min() > 180:
            lon = np.where(lon < 0, lon + 360, lon)
            rings.append(np.column_stack(equal_earth(lon - 360, lat)))
            owners.append(i)
        rings.append(np.column_stack(equal_earth(lon, lat)))
        owners.append(i)
    return rings, np.array(owners, dtype=np.int64)


@cache
def land_cells(resolution: int = DISPLAY_RESOLUTION) -> np.ndarray:
    """Cells whose centre is on land (south of the maps' range left out), sorted."""
    cells: set[int] = set()
    for feature in json.loads(LAND.read_text())["features"]:
        if feature["bbox"][3] >= LATITUDES[0]:
            cells.update(h3.str_to_int(c) for c in h3.geo_to_cells(feature["geometry"], resolution))
    return np.array(sorted(cells), dtype=np.uint64)


def log_edges(top: float, levels: int) -> list[int]:
    """Integer bin edges from 1 to past ``top``, spaced evenly in log, at most ``levels`` bins."""
    ratio = max(top, 1) ** (1 / levels)
    last = int(max(top, 1)) + 1
    return [*sorted({e for i in range(levels) if (e := round(ratio**i)) < last}), last]


_UNITS = ((10**9, "B"), (10**6, "M"), (10**3, "K"))


def short(n: int) -> str:
    """3 significant digits past a thousand: ``999``, ``1.91K``, ``12.6K``, ``2M``."""
    return next((f"{n / size:.3g}{unit}" for size, unit in _UNITS if n >= size), str(n))


def edge_labels(edges: list[int]) -> list[str]:
    """``1``, ``2-3``, ...: the integers each bin [lo, hi) holds."""
    return [
        short(lo) if hi - lo == 1 else f"{short(lo)}-{short(hi - 1)}" for lo, hi in itertools.pairwise(edges)
    ]
