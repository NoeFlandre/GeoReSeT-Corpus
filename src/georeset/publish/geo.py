"""Numbers behind the maps: H3 binning, cell centres, log-spaced colour bins."""

import itertools

import h3
import polars as pl


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
