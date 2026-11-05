import json

import h3.api.basic_int as h3_int
import numpy as np
import polars as pl

from georeset.publish import space, themes

PARIS = (48.85, 2.35)
CAPE = (-33.9, 18.4)


def _sibling(lat: float, lon: float) -> tuple[float, float]:
    """The centre of another resolution-5 cell under the same resolution-4 cell."""
    cell = h3_int.latlng_to_cell(lat, lon, 5)
    parent = h3_int.cell_to_parent(cell, 4)
    other = next(c for c in sorted(h3_int.cell_to_children(parent, 5)) if c != cell)
    return h3_int.cell_to_latlng(other)


def _data() -> pl.LazyFrame:
    near = _sibling(*PARIS)
    rows = [
        ("a", "Fields and woods along the river, then vineyards.", True, "fra_Latn", "website", "u1", *PARIS),
        ("b", "A bakery.", False, "fra_Latn", "wikipedia", "u2", *PARIS),
        (
            "c",
            "Vineyards on the south-facing slopes of the hills.",
            True,
            "fra_Latn",
            "osm_description",
            None,
            *PARIS,
        ),
        ("d", "Forest edge.", True, "deu_Latn", "wikivoyage", "u4", *near),
        ("e", "Dunes.", True, "eng_Latn", "website", "u5", *CAPE),
    ]
    schema = ["id", "text", "landuse", "language", "source", "url", "lat", "lon"]
    return pl.DataFrame(rows, schema=schema, orient="row").lazy()


def _photos() -> pl.LazyFrame:
    return pl.LazyFrame(
        {"scene_id": [1, 1, 2], "lat": [PARIS[0]] * 2 + [CAPE[0]], "lon": [PARIS[1]] * 2 + [CAPE[1]]}
    )


def _scenes() -> pl.LazyFrame:
    return pl.LazyFrame(
        {"scene_id": [1, 2], "text": ["Fields.", "Sand."], "lulc_label": ["Cropland", "Bare"]}
    )


def _work(tmp_path):
    work = tmp_path / "work"
    folder = work / "themes" / "landuse"
    folder.mkdir(parents=True)
    paris = h3_int.latlng_to_cell(*PARIS, 5)
    near = h3_int.latlng_to_cell(*_sibling(*PARIS), 5)
    cells = pl.DataFrame(
        {
            "cell": pl.Series([paris, near], dtype=pl.UInt64),
            "n": [6, 2],
            "pc1": [1.0, -1.0],
            "pc2": [0.0, 0.0],
            "pc3": [0.5, 0.5],
            "l1": [0.4, 0.0],
            "l2": [-0.2, 0.0],
        }
    )
    cells.write_parquet(folder / "cells.parquet")
    display = {"hexagon": [1, 2], "pc1": [-2.0, 2.0], "pc2": [-1.0, 1.0], "pc3": [-1.0, 1.0]}
    pl.DataFrame({**display, "l1": [-0.5, 0.5], "l2": [-0.5, 0.5]}).write_parquet(folder / "hexagons.parquet")
    summary = {"sentences": 8, "explained": [0.3, 0.2, 0.1], "local_explained": [0.2, 0.1]}
    (folder / "summary.json").write_text(json.dumps(summary))
    return work


def test_sentence_cells_count_per_cell_and_language():
    counts, languages = space.sentence_cells(_data())
    paris = h3_int.latlng_to_cell(*PARIS, 5)
    row = counts.filter(cell=paris).row(0, named=True)
    # Relevant sentences only: the bakery (Wikipedia, not relevant) is left out.
    assert row == {
        "cell": paris,
        "relevant": 2,
        "website": 1,
        "description": 1,
        "wikipedia": 0,
        "wikivoyage": 0,
    }
    assert languages.filter(cell=paris)["n"].sum() == 2


def test_build_writes_every_resolution_samples_and_manifest(tmp_path):
    commits = {"data": "c1", "scenes": "c2"}
    out = space.build(
        _data(), _photos(), _scenes(), _work(tmp_path), tmp_path / "out", "me/data", "abc123", commits
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["revision"] == "abc123" and manifest["dataset"] == "me/data"
    assert manifest["folders"] == commits
    assert manifest["resolutions"] == [2, 3, 4, 5]
    assert manifest["languages"][0] == "fra_Latn"
    assert manifest["pca"]["landuse"]["axes"] == [list(themes.LANDUSE.axes[i]) for i in (1, 2, 3)]
    r4 = pl.read_parquet(out / "r4.parquet")
    paris4 = format(h3_int.latlng_to_cell(*PARIS, 4), "x")
    row = r4.filter(cell=paris4).row(0, named=True)
    # Two resolution-5 cells: 2 and 1 relevant sentences; the colour is their mean, each cell counting once.
    assert row["relevant"] == 3 and row["children"] == 2 and row["m_relevant"] == 1.5
    assert row["wikivoyage"] == 1 and row["m_wikivoyage"] == 1.0
    assert row["language"] == "fra_Latn"
    assert r4.schema["relevant"] == pl.Int32 and r4.schema["m_relevant"] == pl.Float32
    assert "sentences" not in r4.columns and "share" not in r4.columns
    # A count no child holds has no mean: empty, never NaN (Cape Town has no Wikipedia sentence).
    cape = r4.filter(cell=format(h3_int.latlng_to_cell(*CAPE, 4), "x")).row(0, named=True)
    assert cape["m_wikipedia"] is None and cape["m_photos"] == 1.0
    assert r4["m_wikipedia"].is_nan().sum() == 0
    # PCA: the two cells' mean scores (0, 0, 0.5) scaled between the display limits (2nd and 98th
    # percentiles: ±1.92, ±0.96, ±0.96), packed as r << 16 | g << 8 | b.
    packed = row["pca_landuse"]
    assert (packed >> 16 & 255, packed >> 8 & 255, packed & 255) == (128, 128, 194)
    assert manifest["chunks"] and all((out / "r5" / f"{c}.parquet").is_file() for c in manifest["chunks"])
    r5 = pl.concat(pl.read_parquet(p) for p in (out / "r5").glob("*.parquet"))
    near = r5.filter(cell=format(h3_int.latlng_to_cell(*_sibling(*PARIS), 5), "x"))
    assert near["pca_landuse"].null_count() == 1  # 2 texts: under PCA_MIN
    samples = pl.concat(pl.read_parquet(p) for p in (out / "samples").glob("*.parquet"))
    paris5 = format(h3_int.latlng_to_cell(*PARIS, 5), "x")
    here = samples.filter(cell=paris5)
    assert sorted(here["text"].to_list()) == [
        "Fields and woods along the river, then vineyards.",
        "Vineyards on the south-facing slopes of the hills.",
    ]
    assert here["scene"].unique().to_list() == ["Fields."]


def test_sentence_samples_keep_a_few_per_cell_and_source():
    rows = [
        (
            f"w{i}",
            f"A long enough sentence about the fields, number {i}.",
            True,
            "fra_Latn",
            "website",
            None,
            *PARIS,
        )
        for i in range(5)
    ]
    rows.append(
        (
            "k",
            "A long enough sentence about the forest of the region.",
            True,
            "fra_Latn",
            "wikipedia",
            None,
            *PARIS,
        )
    )
    schema = ["id", "text", "landuse", "language", "source", "url", "lat", "lon"]
    picked = space.sentence_samples(pl.DataFrame(rows, schema=schema, orient="row").lazy())
    assert picked.group_by("source").len().sort("source").rows() == [
        ("website", space.SAMPLES),
        ("wikipedia", 1),
    ]


def test_rollup_keeps_cells_without_pca(tmp_path):
    cells, languages = space.sentence_cells(_data())
    cells = cells.with_columns(photos=pl.lit(0, pl.Int64))
    table = space.rollup(cells, languages, {}, 3)
    assert table.height == 2 and "pca_landuse" not in table.columns


def test_copy_app(tmp_path):
    space.copy_app(tmp_path / "s")
    names = sorted(p.name for p in (tmp_path / "s").iterdir())
    assert names == ["README.md", "app.js", "index.html", "style.css"]
    assert "sdk: static" in (tmp_path / "s" / "README.md").read_text()


def test_colours_pack_rgb():
    scores = pl.DataFrame(
        {
            "cell": [1, 2],
            "n": [9, 1],
            "pc1": [1.0, 0.0],
            "pc2": [-1.0, 0.0],
            "pc3": [0.0, 0.0],
            "l1": [0.0, 0.0],
            "l2": [-2.0, 0.0],
        }
    )
    limits = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    out = space._colours("x", scores, limits, np.array([1.0, 1.0]))
    assert out.columns == ["cell", "x", "x_local", "x_axes"]
    # One cell with enough texts; its global colour, and its local positions as bytes: 0 and -1 (clipped).
    assert out.rows()[0][:2] == (1, 255 << 16 | 0 << 8 | 128)
    assert out["x_axes"][0] == (0 << 8 | (-127 & 255))
    # The local colour is the global one turned back in hue, as the card draws it.
    rgb = np.array([[1.0, 0.0, 0.5]])
    expected = space._packed(space.colour.nudge(rgb, np.array([[0.0, -1.0]])))[0]
    assert out["x_local"][0] == expected != out["x"][0]
