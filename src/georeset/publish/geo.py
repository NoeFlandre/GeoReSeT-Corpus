"""Numbers behind the maps: H3 binning, cell centres, log-spaced colour bins, the hexagon grid."""

import itertools

import h3
import numpy as np
import polars as pl

HEX_DEGREES = 2
# Rows of the hexagon grid that make regular hexagons over 180° of latitude.
HEX_GRID = (360 // HEX_DEGREES, round(180 / (HEX_DEGREES * np.sqrt(3))))
HEX_EXTENT = (-180, 180, -90, 90)
_NX, _NY = HEX_GRID
_FIRST = (_NX + 1) * (_NY + 1)  # hexagons of the first lattice; the second, offset half a cell, follows
HEXAGONS = _FIRST + _NX * _NY
# matplotlib pads the x range so that points on its edges stay inside.
_X0 = HEX_EXTENT[0] - 1e-9 * (HEX_EXTENT[1] - HEX_EXTENT[0])
_SX = (HEX_EXTENT[1] - HEX_EXTENT[0]) * (1 + 2e-9) / _NX
_Y0 = HEX_EXTENT[2]
_SY = (HEX_EXTENT[3] - HEX_EXTENT[2]) / _NY


def hex_counts(points: pl.DataFrame, resolution: int) -> pl.DataFrame:
    """Sum ``n`` per H3 cell of each (lat, lon) point."""
    cells = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in points.select("lat", "lon").iter_rows()]
    return points.with_columns(cell=pl.Series(cells, dtype=pl.String)).group_by("cell").agg(pl.col("n").sum())


def cell_centres(counts: pl.DataFrame) -> pl.DataFrame:
    """(lat, lon, n) of each cell's centre."""
    centres = [h3.cell_to_latlng(c) for c in counts["cell"]]
    return pl.DataFrame(
        {"lat": [lat for lat, _ in centres], "lon": [lon for _, lon in centres], "n": counts["n"]},
        schema={"lat": pl.Float64, "lon": pl.Float64, "n": pl.Int64},
    )


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


def hexagons(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Index of each point's hexagon in ``Axes.hexbin``'s grid (``HEX_GRID`` over ``HEX_EXTENT``), -1 outside.

    Same rule as matplotlib: the nearer centre of two interleaved rectangular lattices.
    """
    ix = (np.asarray(lon, float) - _X0) / _SX
    iy = (np.asarray(lat, float) - _Y0) / _SY
    ix1, iy1 = np.round(ix).astype(int), np.round(iy).astype(int)
    ix2, iy2 = np.floor(ix).astype(int), np.floor(iy).astype(int)
    first = (ix - ix1) ** 2 + 3 * (iy - iy1) ** 2 < (ix - ix2 - 0.5) ** 2 + 3 * (iy - iy2 - 0.5) ** 2
    inside1 = (ix1 >= 0) & (ix1 <= _NX) & (iy1 >= 0) & (iy1 <= _NY)
    inside2 = (ix2 >= 0) & (ix2 < _NX) & (iy2 >= 0) & (iy2 < _NY)
    return np.where(
        first,
        np.where(inside1, ix1 * (_NY + 1) + iy1, -1),
        np.where(inside2, _FIRST + ix2 * _NY + iy2, -1),
    )


def hexagon_centres(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of each hexagon's centre: drawn with ``hexbin``, a centre fills exactly its hexagon."""
    index = np.asarray(index)
    second = index - _FIRST
    first = second < 0
    column = np.where(first, index // (_NY + 1), second // _NY + 0.5)
    row = np.where(first, index % (_NY + 1), second % _NY + 0.5)
    return _X0 + column * _SX, _Y0 + row * _SY
