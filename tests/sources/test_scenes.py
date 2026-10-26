import math

import polars as pl
import pytest
from conftest import SCENE_TEXT

from georeset.sources import scenes


def _scan(raw, name):
    return pl.scan_parquet(raw / name)


def test_scenes_drop_failed_fix_labels_and_blank_descriptions(raw_scenes):
    out = scenes.scenes(_scan(raw_scenes, scenes.CLUSTERS_FILE)).collect().sort("scene_id")
    assert out.schema == scenes.SCENE_SCHEMA
    assert out.rows() == [
        (3, "Dry bare earth.", None, "Barren soil / Badlands", 0, "Arid shrubland"),
        (7, SCENE_TEXT, "street", "Transportation network", 1, "High-density built-up"),
    ]


def test_scenes_drop_failed_label_even_with_text():
    clusters = pl.LazyFrame(
        {
            "cluster_id": [1, 2],
            "cluster_description": ["a", "b"],
            "visual_description": ["v", "w"],
            "cluster_label": [scenes.FAILED_LABEL, "Lake / Pond"],
            "parent_cluster_id": [0, 0],
            "parent_cluster_label": ["Coastal settlement", "Lake / Pond"],
        }
    )
    out = scenes.scenes(clusters).collect()
    assert out["scene_id"].to_list() == [2]


def test_parent_labels_are_fixed_too():
    clusters = pl.LazyFrame(
        {
            "cluster_id": [1],
            "cluster_description": ["a"],
            "visual_description": ["v"],
            "cluster_label": ["Coastal settlement / Harbor"],
            "parent_cluster_id": [0],
            "parent_cluster_label": ["Bare rock / Gravel plains"],
        }
    )
    row = scenes.scenes(clusters).collect().row(0, named=True)
    assert (row["lulc_label"], row["parent_lulc_label"]) == (
        "Other man-made surface",
        "Rocky desert / Gravel plains",
    )


def test_continent():
    df = pl.DataFrame({"c": ["North America", "Europe", "Ocean / Unknown", None]})
    assert df.select(x=scenes.continent(pl.col("c")))["x"].to_list() == [
        "north-america",
        "europe",
        None,
        None,
    ]


@pytest.mark.parametrize(
    ("platform", "image_url", "license_", "expected"),
    [
        ("mapillary", "mapillary://12", "CC BY-SA 4.0", "https://www.mapillary.com/app/?pKey=12"),
        ("flickr", "https://f/12.jpg", "CC BY 2.0", "https://f/12.jpg"),
        ("flickr", "https://f/12.jpg", "All Rights Reserved", None),
        ("flickr", "https://f/12.jpg", "Do not reuse", None),
        ("flickr", "https://f/12.jpg", None, None),
        ("mapillary", "mapillary://12", None, None),
        ("wikimedia", "/home/x/12.jpg", "CC BY SA 3.0", None),
        ("kartaview", "kartaview://12", "CC BY-SA 4.0", None),
        ("flickr", "http://f/12.jpg", "CC BY 2.0", None),
    ],
)
def test_url(platform, image_url, license_, expected):
    df = pl.DataFrame(
        {"p": [platform], "u": [image_url], "i": ["12"], "l": [license_]},
        schema=dict.fromkeys("puil", pl.String),
    )
    out = df.select(scenes.url(pl.col("p"), pl.col("u"), pl.col("i"), pl.col("l")))
    assert out.item() == expected


def test_photos(raw_scenes):
    out = (
        scenes.photos(_scan(raw_scenes, scenes.PHOTOS_FILE), _scan(raw_scenes, scenes.CLUSTERS_FILE))
        .collect()
        .sort("photo_id")
    )
    assert out.schema == scenes.PHOTO_SCHEMA
    first = out.row(0, named=True)
    assert first == {
        "scene_id": 7,
        "platform": "mapillary",
        "photo_id": "1",
        "url": "https://www.mapillary.com/app/?pKey=1",
        "photo_license": "CC BY-SA 4.0",
        "captured_at": "2020-01-01T00:00:00Z",
        "season": "Winter",
        "time_of_day": "Night",
        "koppen": "Cfb",
        "country": "France",
        "continent": "europe",
        "h3_cell": "8b1",
        "lat": 48.0,
        "lon": 2.0,
    }
    assert out["season"].to_list() == ["Winter", None, "Wet Season", "Summer"]
    assert out["time_of_day"].to_list() == ["Night", "Morning", None, "Dusk"]
    assert out["url"].to_list() == [first["url"], "https://live.staticflickr.com/2.jpg", None, None]
    assert out["continent"].to_list() == ["europe", "europe", "north-america", None]


def test_photos_blank_license_is_null():
    meta = pl.LazyFrame(
        {
            "photo_key": ["k"],
            "License": ["  "],
            **{c: ["x"] for c in ["Platform", "Photo_ID", "Image_URL", "Captured_At", "Season"]},
            **{c: ["x"] for c in ["Time_Of_Day", "Koppen_Code", "country", "continent", "H3_Cell"]},
            "Latitude": [0.0],
            "Longitude": [0.0],
        }
    )
    out = scenes.photos(meta, pl.LazyFrame({"photo_key": ["k"], "cluster_id": [1]})).collect()
    assert out["photo_license"].to_list() == [None]


def test_haversine_km():
    df = pl.DataFrame(
        {
            "a": [0.0, 0.0, 48.8566],
            "b": [0.0, 0.0, 2.3522],
            "c": [0.0, 1.0, 51.5074],
            "d": [0.0, 0.0, -0.1278],
        }
    )
    out = df.select(x=scenes.haversine_km(pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d")))["x"].to_list()
    assert out[0] == 0
    assert out[1] == pytest.approx(math.pi * scenes.EARTH_RADIUS_KM / 180)
    assert out[2] == pytest.approx(343.5, abs=0.5)  # Paris - London


def test_spread():
    photos = pl.LazyFrame(
        {
            "scene_id": [1, 1, 1, 1, 2],
            "lat": [0.0, 0.0, 0.0, 0.0, 5.0],
            "lon": [0.0, 1.0, 2.0, 10.0, 5.0],
            "country": ["C", "B", "B", "A", "D"],
        }
    )
    out = scenes.spread(photos).collect().sort("scene_id")
    one = out.row(0, named=True)
    degree = math.pi * scenes.EARTH_RADIUS_KM / 180
    assert one["n_photos"] == 4
    assert (one["lat"], one["lon"]) == (0.0, 1.5)
    assert one["spread_km_p50"] == pytest.approx(degree)  # distances 1.5, 0.5, 0.5, 8.5 degrees
    assert one["spread_km_p90"] == pytest.approx(degree * (1.5 + 0.7 * 7))
    assert one["countries"] == ["B", "A", "C"]
    two = out.row(1, named=True)
    assert (two["n_photos"], two["spread_km_p50"], two["spread_km_p90"], two["countries"]) == (1, 0, 0, ["D"])


def test_with_spread_keeps_scenes_with_photos_sorted():
    scene_rows = pl.LazyFrame({"scene_id": [5, 2, 9], "text": ["e", "b", "i"]})
    photos = pl.LazyFrame({"scene_id": [5, 2], "lat": [0.0, 1.0], "lon": [0.0, 1.0], "country": ["A", "B"]})
    out = scenes.with_spread(scene_rows, photos).collect()
    assert out["scene_id"].to_list() == [2, 5]
    assert out.columns[:3] == ["scene_id", "text", "n_photos"]
