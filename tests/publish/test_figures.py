import h3
import polars as pl
import shapely

from georeset.publish import figures


def test_world_map_writes_png(tmp_path):
    cells = [h3.latlng_to_cell(48.85, 2.35, 3), h3.latlng_to_cell(0.0, 179.99, 1)]
    counts = pl.DataFrame({"cell": cells, "n": [10, 5]})
    path = figures.world_map(counts, [shapely.box(-10, 35, 30, 60)], tmp_path / "a" / "map.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_language_bars_writes_png(tmp_path):
    top = pl.DataFrame({"language": ["fra_Latn", "eng_Latn"], "sentences": [10, 5]})
    path = figures.language_bars(top, tmp_path / "languages.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
