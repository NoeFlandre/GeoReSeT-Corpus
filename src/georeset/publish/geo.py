"""Numbers behind the maps: H3 binning, cell centres, log-spaced colour bins, the hexagon grid."""

import itertools
from dataclasses import dataclass
from functools import cache, cached_property
from importlib.resources import files

import h3
import numpy as np
import polars as pl
import shapely

HEX_DEGREES = 2
HEX_EXTENT = (-180, 180, -90, 90)
# Corners of a hexbin hexagon around its centre, in steps of its grid, as matplotlib draws them.
_CORNERS = np.array([[0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-0.5, -0.5], [0.0, -1.0]]) * [
    1,
    1 / 3,
]
# Natural Earth 1:110m land (public domain, v5.1.2).
LAND = files("georeset.publish").joinpath("assets/ne_110m_land.geojson")


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


@dataclass(frozen=True)
class Grid:
    """``Axes.hexbin``'s grid of ``degrees``-wide hexagons over ``HEX_EXTENT``.

    Hexagons are numbered as matplotlib counts them: the first lattice, then the second, offset half a cell.
    """

    degrees: float

    @cached_property
    def size(self) -> tuple[int, int]:
        """``gridsize``: columns, and rows that make regular hexagons over 180° of latitude."""
        return round(360 / self.degrees), round(180 / (self.degrees * np.sqrt(3)))

    @cached_property
    def count(self) -> int:
        nx, ny = self.size
        return (nx + 1) * (ny + 1) + nx * ny

    @cached_property
    def _frame(self) -> tuple[float, float, float, float]:
        """x0, x step, y0, y step; matplotlib pads the x range so that points on its edges stay inside."""
        nx, ny = self.size
        width = HEX_EXTENT[1] - HEX_EXTENT[0]
        return (
            HEX_EXTENT[0] - 1e-9 * width,
            width * (1 + 2e-9) / nx,
            HEX_EXTENT[2],
            (HEX_EXTENT[3] - HEX_EXTENT[2]) / ny,
        )

    def hexagons(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Index of each point's hexagon, -1 outside.

        Same rule as matplotlib: the nearer centre of two interleaved rectangular lattices.
        """
        nx, ny = self.size
        x0, sx, y0, sy = self._frame
        ix = (np.asarray(lon, float) - x0) / sx
        iy = (np.asarray(lat, float) - y0) / sy
        ix1, iy1 = np.round(ix).astype(int), np.round(iy).astype(int)
        ix2, iy2 = np.floor(ix).astype(int), np.floor(iy).astype(int)
        first = (ix - ix1) ** 2 + 3 * (iy - iy1) ** 2 < (ix - ix2 - 0.5) ** 2 + 3 * (iy - iy2 - 0.5) ** 2
        inside1 = (ix1 >= 0) & (ix1 <= nx) & (iy1 >= 0) & (iy1 <= ny)
        inside2 = (ix2 >= 0) & (ix2 < nx) & (iy2 >= 0) & (iy2 < ny)
        return np.where(
            first,
            np.where(inside1, ix1 * (ny + 1) + iy1, -1),
            np.where(inside2, (nx + 1) * (ny + 1) + ix2 * ny + iy2, -1),
        )

    def centres(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) of each hexagon's centre: drawn with ``hexbin``, a centre fills exactly its hexagon."""
        nx, ny = self.size
        x0, sx, y0, sy = self._frame
        index = np.asarray(index)
        second = index - (nx + 1) * (ny + 1)
        first = second < 0
        column = np.where(first, index // (ny + 1), second // ny + 0.5)
        row = np.where(first, index % (ny + 1), second % ny + 0.5)
        return x0 + column * sx, y0 + row * sy

    def shapes(self, index: np.ndarray) -> np.ndarray:
        """Each hexagon as a shapely polygon, the shape ``hexbin`` draws."""
        _, sx, _, sy = self._frame
        lon, lat = self.centres(index)
        return shapely.polygons(np.stack([lon, lat], axis=1)[:, None, :] + _CORNERS * [sx, sy])

    def land(self) -> np.ndarray:
        """Indices of the hexagons that touch land."""
        return _land(self)


@cache
def _land(grid: Grid) -> np.ndarray:
    land = shapely.from_geojson(LAND.read_text())
    shapely.prepare(land)
    return np.flatnonzero(shapely.intersects(land, grid.shapes(np.arange(grid.count))))


COARSE = Grid(HEX_DEGREES)  # the grid of the card's maps
FINE = Grid(0.25)  # the grid of the fine RGB maps
