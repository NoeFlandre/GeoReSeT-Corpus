import h3
import numpy as np
import polars as pl
import pytest
import shapely

from georeset.publish import geo

PARIS = h3.latlng_to_cell(48.85, 2.35, 3)


def test_hex_counts_sums_points_per_cell():
    pts = pl.DataFrame({"lat": [48.85, 48.8501, -33.9], "lon": [2.35, 2.3501, 18.4], "n": [2, 3, 7]})
    out = geo.hex_counts(pts, 3).sort("n")
    assert out.rows() == [(PARIS, 5), (h3.latlng_to_cell(-33.9, 18.4, 3), 7)]


def test_hex_counts_resolution_matters():
    pts = pl.DataFrame({"lat": [48.85], "lon": [2.35], "n": [1]})
    assert geo.hex_counts(pts, 5)["cell"].item() == h3.latlng_to_cell(48.85, 2.35, 5)


def test_hex_counts_of_no_points():
    pts = pl.DataFrame(
        {"lat": [], "lon": [], "n": []}, schema={"lat": pl.Float64, "lon": pl.Float64, "n": pl.Int64}
    )
    assert geo.hex_counts(pts, 5).height == 0


def test_cell_centres():
    out = geo.cell_centres(pl.DataFrame({"cell": [PARIS], "n": [4]}))
    lat, lon = h3.cell_to_latlng(PARIS)
    assert out.rows() == [(lat, lon, 4)]
    assert abs(lat - 48.85) < 1 and abs(lon - 2.35) < 1


def test_cell_centres_of_nothing():
    out = geo.cell_centres(pl.DataFrame({"cell": [], "n": []}, schema={"cell": pl.String, "n": pl.Int64}))
    assert out.schema == {"lat": pl.Float64, "lon": pl.Float64, "n": pl.Int64}
    assert out.height == 0


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


def _hexbin_cells(lon, lat):
    """Index of the hexagon matplotlib's own ``hexbin`` puts each point in."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cells = []
    for x, y in zip(lon, lat, strict=True):
        hexes = ax.hexbin([x], [y], gridsize=geo.COARSE.size, extent=geo.HEX_EXTENT, mincnt=1)
        lon_c, lat_c = hexes.get_offsets()[0]
        cells.append((round(float(lon_c), 6), round(float(lat_c), 6)))
        hexes.remove()
    plt.close(fig)
    return cells


def test_hexagons_match_matplotlib_hexbin():
    rng = np.random.default_rng(0)
    lon = np.concatenate([rng.uniform(-180, 180, 300), [-180.0, 180.0, 0.0]])
    lat = np.concatenate([rng.uniform(-89, 89, 300), [0.0, 0.0, -90.0]])
    index = geo.COARSE.hexagons(lon, lat)
    assert (index >= 0).all()
    assert (index < geo.COARSE.count).all()
    centre_lon, centre_lat = geo.COARSE.centres(index)
    ours = [(round(float(x), 6), round(float(y), 6)) for x, y in zip(centre_lon, centre_lat, strict=True)]
    assert ours == _hexbin_cells(lon, lat)


def test_hexagons_both_lattices_and_outside():
    centre_lon, centre_lat = geo.COARSE.centres(np.array([0, geo.COARSE.count - 1]))
    assert geo.COARSE.hexagons(centre_lon, centre_lat).tolist() == [0, geo.COARSE.count - 1]
    assert geo.COARSE.hexagons(np.array([200.0, 0.0]), np.array([0.0, 120.0])).tolist() == [-1, -1]


def test_hexagon_grid_is_two_degrees():
    assert geo.COARSE.size == (180, 52)
    lon, lat = geo.COARSE.centres(np.array([0, 1, 53]))
    assert lon[1] == lon[0] and lat[1] - lat[0] == pytest.approx(180 / 52)
    assert lon[2] - lon[0] == pytest.approx(2)


def test_hexagon_shapes_are_the_hexagons_around_their_centres():
    index = np.array([0, 5, geo.COARSE.count - 1])
    shapes = geo.COARSE.shapes(index)
    lon, lat = geo.COARSE.centres(index)
    centroids = shapely.centroid(shapes)
    np.testing.assert_allclose(shapely.get_x(centroids), lon)
    np.testing.assert_allclose(shapely.get_y(centroids), lat)
    # Two interleaved lattices of hexagons tile the plane: each covers half a lattice cell.
    cell = 360 / geo.COARSE.size[0] * 180 / geo.COARSE.size[1]
    np.testing.assert_allclose(shapely.area(shapes), cell / 2)


def test_land_hexagons():
    land = geo.COARSE.land()
    paris, pacific = geo.COARSE.hexagons(np.array([2.35, -140.0]), np.array([48.85, -10.0]))
    assert paris in land and pacific not in land
    assert 0.2 < len(land) / geo.COARSE.count < 0.5
    assert np.all(np.diff(land) > 0)
