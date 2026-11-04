import h3
import polars as pl

from georeset.publish import figures


def _counts(cells: list[str], n: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"cell": cells, "n": n}, schema={"cell": pl.String, "n": pl.Int64})


def test_world_map_writes_png(tmp_path):
    counts = _counts([h3.latlng_to_cell(48.85, 2.35, 5), h3.latlng_to_cell(0.0, 179.99, 5)], [10, 5])
    full = _counts([h3.latlng_to_cell(-33.9, 18.4, 5)], [3])
    path = figures.world_map(counts, tmp_path / "a" / "map.png", "Sentences", 5, full)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_world_map_draws_an_empty_subset(tmp_path):
    path = figures.world_map(_counts([], []), tmp_path / "map.png", "Photos", 5)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_language_bars_writes_png(tmp_path):
    top = pl.DataFrame({"language": ["fra_Latn", "eng_Latn"], "sentences": [10, 5]})
    path = figures.language_bars(top, tmp_path / "languages.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
