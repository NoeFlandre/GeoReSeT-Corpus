import h3.api.basic_int as h3_int
import numpy as np
import polars as pl

from georeset.publish import figures


def _cells(*points) -> np.ndarray:
    return np.array([h3_int.latlng_to_cell(lat, lon, 4) for lat, lon in points], np.uint64)


def test_world_map_writes_png(tmp_path):
    cells = _cells((48.85, 2.35), (0.0, 179.99))
    path = figures.world_map(
        cells, np.array([10.0, 5.0]), tmp_path / "a" / "map.png", "Sentences", _cells((-33.9, 18.4))
    )
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_world_map_draws_an_empty_subset(tmp_path):
    path = figures.world_map(np.array([], np.uint64), np.array([]), tmp_path / "map.png", "Photos")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_pca_maps_write_png(tmp_path):
    cells, land = _cells((48.85, 2.35), (-33.9, 18.4)), _cells((40.0, -100.0))
    rgb = figures.rgb_map(
        cells, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]), tmp_path / "rgb.png", land, "x"
    )
    pc = figures.diverging_map(cells, np.array([-1.0, 2.0]), 1.5, "PC1", tmp_path / "pc.png", land)
    assert rgb.read_bytes()[:4] == pc.read_bytes()[:4] == b"\x89PNG"


def test_language_bars_writes_png(tmp_path):
    top = pl.DataFrame({"language": ["fra_Latn", "eng_Latn"], "sentences": [10, 5]})
    path = figures.language_bars(top, tmp_path / "languages.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
