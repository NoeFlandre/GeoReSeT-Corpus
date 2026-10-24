"""Geometry behind the map: H3 binning, projected rings, Natural Earth land."""

import io
import urllib.request
import zipfile
from pathlib import Path

import h3
import numpy as np
import polars as pl
import shapefile
import shapely
from pyproj import Transformer

LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
LAND_SHAPES = "ne_110m_land.shp"
_PROJECT = Transformer.from_crs("EPSG:4326", "+proj=eqearth", always_xy=True)


def hex_counts(points: pl.DataFrame, resolution: int) -> pl.DataFrame:
    """Sum ``n`` per H3 cell of each (lat, lon) point."""
    cells = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in points.select("lat", "lon").iter_rows()]
    return points.with_columns(cell=pl.Series(cells)).group_by("cell").agg(pl.col("n").sum())


def cell_ring(cell: str) -> np.ndarray | None:
    """Projected boundary of a cell; ``None`` for the few crossing the antimeridian."""
    lat, lon = np.array(h3.cell_to_boundary(cell)).T
    if lon.max() - lon.min() > 180:
        return None
    return np.column_stack(_PROJECT.transform(lon, lat))


def bin_index(n: pl.Series, levels: int) -> pl.Series:
    """Decade bin of each count (1-9 -> 0, 10-99 -> 1, ...), capped at ``levels - 1``."""
    return n.log10().floor().clip(upper_bound=levels - 1).cast(pl.Int64)


def binned_rings(counts: pl.DataFrame, levels: int) -> list[tuple[np.ndarray, int]]:
    """(projected ring, decade bin) of every drawable cell."""
    binned = counts.select("cell", bin=bin_index(counts["n"], levels))
    return [(ring, b) for c, b in binned.iter_rows() if (ring := cell_ring(c)) is not None]


def load_land(cache: Path) -> list[shapely.Geometry]:
    """Natural Earth 1:110m land polygons, downloaded once into ``cache``."""
    path = cache / "ne_110m_land.zip"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)  # pragma: no mutate
        with urllib.request.urlopen(LAND_URL) as response:
            path.write_bytes(response.read())
    with zipfile.ZipFile(path) as archive:
        shapes = shapefile.Reader(shp=io.BytesIO(archive.read(LAND_SHAPES))).shapes()
        return [shapely.geometry.shape(s.__geo_interface__) for s in shapes if s is not None]


def land_rings(land: list[shapely.Geometry]) -> list[np.ndarray]:
    """Projected exterior ring of every polygon, multipolygons flattened."""
    polygons = [p for geom in land for p in getattr(geom, "geoms", [geom])]
    return [np.column_stack(_PROJECT.transform(*p.exterior.xy)) for p in polygons]
