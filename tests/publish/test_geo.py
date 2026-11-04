import h3
import polars as pl
import pytest

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
    assert geo.edge_labels([1, 2, 4, 1000, 2001]) == ["1", "2-3", "4-999", "1,000-2,000"]
