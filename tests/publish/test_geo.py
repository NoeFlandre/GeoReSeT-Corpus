import io
import zipfile

import h3
import numpy as np
import polars as pl
import pytest
import shapefile
import shapely

from georeset.publish import geo

PARIS = h3.latlng_to_cell(48.85, 2.35, 3)
ANTIMERIDIAN = h3.latlng_to_cell(0.0, 179.99, 1)


def test_hex_counts_sums_points_per_cell():
    pts = pl.DataFrame({"lat": [48.85, 48.8501, -33.9], "lon": [2.35, 2.3501, 18.4], "n": [2, 3, 7]})
    out = geo.hex_counts(pts, 3).sort("n")
    assert out.rows() == [(PARIS, 5), (h3.latlng_to_cell(-33.9, 18.4, 3), 7)]


def test_hex_counts_resolution_matters():
    pts = pl.DataFrame({"lat": [48.85], "lon": [2.35], "n": [1]})
    assert geo.hex_counts(pts, 5)["cell"].item() == h3.latlng_to_cell(48.85, 2.35, 5)


def test_cell_ring_projects_to_equal_earth_metres():
    ring = geo.cell_ring(PARIS)
    assert ring is not None
    assert ring.shape == (6, 2)
    x, y = ring.mean(axis=0)
    assert 150_000 < x < 250_000  # 2.35°E in Equal Earth metres
    assert 5_000_000 < y < 6_000_000


def test_cell_ring_skips_antimeridian_cells():
    assert geo.cell_ring(ANTIMERIDIAN) is None


def test_cell_ring_keeps_cells_just_under_180_degrees_wide(monkeypatch):
    boundary = ((0.0, -90.0), (1.0, 90.0), (0.0, 0.0))  # lon span exactly 180
    monkeypatch.setattr(geo.h3, "cell_to_boundary", lambda cell: boundary)
    assert geo.cell_ring("x") is not None


@pytest.mark.parametrize(("levels", "expected"), [(7, [0, 0, 1, 4, 6]), (3, [0, 0, 1, 2, 2])])
def test_bin_index(levels, expected):
    assert geo.bin_index(pl.Series([1, 9, 10, 99_999, 10**9]), levels).to_list() == expected


def test_binned_rings_pairs_rings_with_bins_and_skips_antimeridian():
    counts = pl.DataFrame({"cell": [ANTIMERIDIAN, PARIS], "n": [5, 1234]})
    [(ring, b)] = geo.binned_rings(counts, 7)
    assert b == 3
    assert np.allclose(ring, geo.cell_ring(PARIS))


def test_land_rings_flattens_multipolygons():
    land = [shapely.box(0, 0, 1, 1), shapely.MultiPolygon([shapely.box(2, 2, 3, 3), shapely.box(4, 4, 5, 5)])]
    rings = geo.land_rings(land)
    assert [r.shape for r in rings] == [(5, 2)] * 3
    assert rings[1][:, 0].min() > rings[0][:, 0].max()


def _land_zip() -> bytes:
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON) as w:
        w.field("id", "N")
        w.poly([[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]])
        w.record(1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in zip(geo.LAND_FILES, (shp, shx, dbf), strict=True):
            z.writestr(name, data.getvalue())
    return buffer.getvalue()


def test_load_land_downloads_once_into_cache(tmp_path, monkeypatch):
    payload, calls = _land_zip(), []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(geo.urllib.request, "urlopen", lambda url: calls.append(url) or Response(payload))
    cache = tmp_path / "nested" / "cache"
    land = geo.load_land(cache)
    assert [g.area for g in land] == [1.0]
    assert (cache / "ne_110m_land.zip").read_bytes() == payload
    assert geo.load_land(cache)[0].equals(land[0])
    assert calls == [geo.LAND_URL]
