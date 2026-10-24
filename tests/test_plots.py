import polars as pl
import shapely

from georeset import plots


def test_hex_counts_sums_points_per_cell():
    pts = pl.DataFrame({"lat": [48.85, 48.8501, -33.9], "lon": [2.35, 2.3501, 18.4], "n": [2, 3, 7]})
    out = plots.hex_counts(pts, 3).sort("n")
    assert out["n"].to_list() == [5, 7]
    assert out.columns == ["cell", "n"]


def test_cell_ring_projects_and_skips_antimeridian():
    import h3

    ring = plots.cell_ring(h3.latlng_to_cell(48.85, 2.35, 3))
    assert ring is not None
    assert ring.shape == (6, 2)
    assert abs(ring[0, 0]) > 1000  # projected metres, not degrees
    assert plots.cell_ring(h3.latlng_to_cell(0.0, 179.99, 1)) is None


def test_bin_index():
    assert plots.bin_index(pl.Series([1, 9, 10, 99_999, 10**9])).to_list() == [0, 0, 1, 4, 6]


def test_language_label():
    assert plots.language_label("fra_Latn") == "French · fra_Latn"
    assert plots.language_label("zzz_Zzzz") == "zzz_Zzzz"


def test_world_map_writes_png(tmp_path):
    import h3

    cells = [h3.latlng_to_cell(48.85, 2.35, 3), h3.latlng_to_cell(0.0, 179.99, 1)]
    counts = pl.DataFrame({"cell": cells, "n": [10, 5]})
    land = [shapely.box(-10, 35, 30, 60)]
    path = plots.world_map(counts, land, tmp_path / "map.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_language_bars_writes_png(tmp_path):
    top = pl.DataFrame({"language": ["fra_Latn", "eng_Latn"], "sentences": [10, 5]})
    path = plots.language_bars(top, tmp_path / "languages.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def _land_zip(tmp_path):
    import io
    import zipfile

    import shapefile

    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON) as w:
        w.field("id", "N")
        w.poly([[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]])
        w.record(1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for ext, data in (("shp", shp), ("shx", shx), ("dbf", dbf)):
            z.writestr(f"ne_110m_land.{ext}", data.getvalue())
    return buffer.getvalue()


def test_load_land_downloads_once(tmp_path, monkeypatch):
    payload = _land_zip(tmp_path)
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(plots.urllib.request, "urlopen", lambda url: calls.append(url) or Response())
    cache = tmp_path / "cache"
    land = plots.load_land(cache)
    assert len(land) == 1
    assert land[0].area == 1.0
    plots.load_land(cache)
    assert calls == [plots.LAND_URL]
