import h3
import h3.api.basic_int as h3_int
import numpy as np
import polars as pl
import pytest

from georeset.publish import geo


def _cell(lat: float, lon: float, resolution: int = 5) -> int:
    return h3_int.latlng_to_cell(lat, lon, resolution)


def test_hex_counts_sums_points_per_cell():
    pts = pl.DataFrame({"lat": [48.85, 48.8501, -33.9], "lon": [2.35, 2.3501, 18.4], "n": [2, 3, 7]})
    out = geo.hex_counts(pts, 3).sort("n")
    assert out.rows() == [(_cell(48.85, 2.35, 3), 5), (_cell(-33.9, 18.4, 3), 7)]
    assert out.schema == {"cell": pl.UInt64, "n": pl.Int64}


def test_hex_counts_defaults_to_the_cell_resolution():
    pts = pl.DataFrame({"lat": [48.85], "lon": [2.35], "n": [1]})
    assert geo.hex_counts(pts)["cell"].item() == _cell(48.85, 2.35, geo.CELL_RESOLUTION)


def test_hex_counts_of_no_points():
    pts = pl.DataFrame(
        {"lat": [], "lon": [], "n": []}, schema={"lat": pl.Float64, "lon": pl.Float64, "n": pl.Int64}
    )
    assert geo.hex_counts(pts).height == 0


def test_h3_cells_parents_and_centres():
    lon, lat = np.array([2.35, 18.4]), np.array([48.85, -33.9])
    cells = geo.h3_cells(lon, lat)
    assert cells.dtype == np.uint64
    assert cells.tolist() == [
        h3.str_to_int(h3.latlng_to_cell(a, o, 5)) for a, o in zip(lat, lon, strict=True)
    ]
    parents = geo.h3_parents(cells)
    assert parents.tolist() == [h3_int.cell_to_parent(int(c), geo.DISPLAY_RESOLUTION) for c in cells]
    centre_lon, centre_lat = geo.h3_centres(cells)
    np.testing.assert_allclose(centre_lon, lon, atol=0.2)
    np.testing.assert_allclose(centre_lat, lat, atol=0.2)
    assert geo.h3_centres(np.array([], np.uint64))[0].size == 0


def test_equal_earth():
    x, y = geo.equal_earth(np.array([0.0, 180.0, -90.0, 0.0]), np.array([0.0, 0.0, 30.0, 90.0]))
    # Published extents on a unit sphere: 2.7066 at the equator's ends, 1.3174 at the pole.
    np.testing.assert_allclose(x, [0.0, 2.70663, -x[2] * -1, 0.0], atol=1e-4)
    np.testing.assert_allclose(y[[0, 1, 3]], [0.0, 0.0, 1.31736], atol=1e-4)
    assert x[2] < 0 < y[2] < y[3]


def test_outline_is_the_closed_edge_of_the_world():
    ring = geo.outline(5)
    assert ring.shape == (10, 2)
    np.testing.assert_allclose(ring[:, 0].max(), 2.70663, atol=1e-4)
    np.testing.assert_allclose(ring[:5, 0], -ring[5:, 0][::-1])


def test_cell_rings_split_cells_across_180():
    paris, fiji = _cell(48.85, 2.35, 4), _cell(-17.0, 180.0, 4)
    rings, owners = geo.cell_rings(np.array([paris, fiji], np.uint64))
    assert owners.tolist() == [0, 1, 1]
    # The two copies of the cell across ±180° lie on either side of the map.
    assert rings[1][:, 0].mean() < 0 < rings[2][:, 0].mean()
    assert all(r.shape == (6, 2) for r in rings)


def test_land_cells():
    land = geo.land_cells()
    assert _cell(48.85, 2.35, 4) in land and _cell(-10.0, -140.0, 4) not in land
    assert _cell(-80.0, 0.0, 4) not in land  # Antarctica is south of the maps
    assert land.dtype == np.uint64 and np.all(np.diff(land.astype(np.int64)) != 0)


@pytest.mark.parametrize(
    ("top", "levels", "expected"),
    [
        (1000, 3, [1, 10, 100, 1001]),
        (999.5, 3, [1, 10, 100, 1000]),
        (5, 11, [1, 2, 3, 4, 6]),
        (1.6, 3, [1, 2]),
        (0.5, 3, [1, 2]),
        (1000, 11, [1, 2, 4, 7, 12, 23, 43, 81, 152, 285, 534, 1001]),
    ],
)
def test_log_edges(top, levels, expected):
    assert geo.log_edges(top, levels) == expected


def test_edge_labels():
    assert geo.edge_labels([1, 2, 4, 1000, 2001]) == ["1", "2-3", "4-999", "1K-2K"]


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1K"),
        (1907, "1.91K"),
        (12_612, "12.6K"),
        (10**6, "1M"),
        (2_500_000_000, "2.5B"),
    ],
)
def test_short(n, expected):
    assert geo.short(n) == expected
