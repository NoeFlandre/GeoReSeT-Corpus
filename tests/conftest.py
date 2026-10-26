from pathlib import Path

import polars as pl
import pytest

from georeset.build.pipeline import Layout, build_meta, build_rows, build_shards
from georeset.build.stage import write_staged
from georeset.sources.base import Staged

SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'


@pytest.fixture
def write(tmp_path: Path):
    """Write a dict of columns to ``tmp_path/name.parquet`` and return the path."""

    def _write(name: str, columns: dict, schema: dict | None = None) -> Path:
        path = tmp_path / f"{name}.parquet"
        pl.DataFrame(columns, schema=schema).write_parquet(path)
        return path

    return _write


def _polygons(ids, name):
    n = len(ids)
    return pl.DataFrame(
        {
            "polygon_id": ids,
            "name": [name] * n,
            "osm_tag": ["landuse=forest"] * n,
            "lat": [1.0] * n,
            "lon": [2.0] * n,
            "area_m2": [3.0] * n,
            "geometry": [b"\x01"] * n,
        }
    )


def _sentences(source, doc, labels, language="fra_Latn"):
    n = len(labels)
    return pl.DataFrame(
        {
            "source": [source] * n,
            "doc_key": [doc] * n,
            "position": list(range(n)),
            "text": [f"{doc} sentence {i}" for i in range(n)],
            "label": labels,
            "failure_reason": [None] * n,
            "language": [language] * n,
            "url": ["u"] * n,
        },
        schema_overrides={"position": pl.Int32},
    )


@pytest.fixture
def layout(tmp_path):
    layout = Layout(tmp_path / "work", tmp_path / "out")
    # The same wiki doc is staged in two overlapping regions; centro is the smaller extract.
    for stem, ids in [
        ("italy-latest", ["way/1", "way/2", "way/3"]),
        ("centro-latest", ["way/1"]),
    ]:
        write_staged(
            Staged(
                polygons=_polygons(ids, stem),
                links=pl.DataFrame({"doc_key": ["Q1"], "polygon_id": ["way/1"]}),
                sentences=_sentences("wikipedia", "Q1", ["yes", "no", "failed"]),
            ),
            layout.stage / "wiki" / stem,
        )
    write_staged(
        Staged(
            polygons=_polygons(["way/2"], "web"),
            links=pl.DataFrame({"doc_key": ["http://x"], "polygon_id": ["way/2"]}),
            sentences=_sentences("website", "http://x", ["no", "yes"], "eng_Latn"),
        ),
        layout.stage / "website" / "italy-latest",
    )
    return layout


def labels_columns(rows: list[tuple], keys: list[str]) -> dict:
    """Upstream label-table columns from ``(keys..., decision, failure_reason, language)`` rows."""
    names = [*keys, "decision", "failure_reason", "language"]
    return {name: [row[i] for row in rows] for i, name in enumerate(names)}


def build_all(layout):
    build_meta(layout)
    for staged in sorted(layout.stage.glob("*/*")):
        build_rows(layout, staged)
    build_shards(layout, workers=2)


SCENE_TEXT = "A paved street lined with trees."


@pytest.fixture
def raw_scenes(tmp_path):
    """Raw geo-rag files: 4 photos in 3 clusters, one of which failed to be described."""
    from georeset.sources import scenes

    folder = tmp_path / "raw"
    folder.mkdir()
    keys = ["mapillary_1", "flickr_2", "wikimedia_3", "flickr_4"]
    pl.DataFrame(
        {
            "Photo_ID": ["1", "2", "3", "4"],
            "Platform": ["mapillary", "flickr", "wikimedia", "flickr"],
            "Latitude": [48.0, 48.0, 10.0, 0.0],
            "Longitude": [2.0, 3.0, 20.0, 0.0],
            "Image_URL": [
                "mapillary://1",
                "https://live.staticflickr.com/2.jpg",
                "/home/someone/landmarks/3.jpg",
                "https://live.staticflickr.com/4.jpg",
            ],
            "Captured_At": ["2020-01-01T00:00:00Z", None, "2019-06-01T12:00:00Z", None],
            "H3_Cell": ["8b1", "8b2", "8b3", "8b4"],
            "Season": ["Winter", "Unknown", "Wet Season", "Summer"],
            "Time_Of_Day": ["Night", "Morning", "Unknown", "Dusk"],
            "Koppen_Code": ["Cfb", "Cfb", "Aw", "Af"],
            "Koppen_Desc": ["d"] * 4,
            "continent": ["Europe", "Europe", "North America", "Ocean / Unknown"],
            "country": ["France", "France", "Mexico", "Ocean"],
            "License": ["CC BY-SA 4.0", "CC BY 2.0", "CC BY SA 3.0", "All Rights Reserved"],
            "photo_key": keys,
        }
    ).write_parquet(folder / scenes.PHOTOS_FILE)
    pl.DataFrame(
        {
            "photo_key": keys,
            "cluster_id": [7, 7, 3, 9],
            "parent_cluster_id": [1, 1, 0, 1],
            "cluster_label": [
                "Transportation network",
                "Transportation network",
                "Bare soil / Badlands",
                "x",
            ],
            "cluster_description": [SCENE_TEXT, SCENE_TEXT, "Dry bare earth.", scenes.FAILED_TEXT],
            "visual_description": ["street", "street", "", "?"],
            "parent_cluster_label": ["High-density built-up"] * 2
            + ["Arid shrubland", "High-density built-up"],
        }
    ).write_parquet(folder / scenes.CLUSTERS_FILE)
    return folder
