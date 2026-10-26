import polars as pl

from georeset.publish import stats


def _data():
    return pl.LazyFrame(
        {
            "text": ["a b c", "d e", "f", "g h"],
            "landuse": [True, False, True, True],
            "language": ["fra_Latn", "fra_Latn", "eng_Latn", "fra_Latn"],
            "source": ["wikipedia", "wikipedia", "website", "website"],
            "polygon_ids": [["way/1"], ["way/1", "way/2"], ["way/3"], ["way/3"]],
            "region": ["italy", "centro", "italy", "italy"],
            "continent": ["europe", "europe", "europe", "africa"],
            "lat": [1.0, 1.0, 2.0, 2.0],
            "lon": [3.0, 3.0, 4.0, 4.0],
        }
    )


def test_by_source_and_total():
    out = stats.by_source(_data())
    assert out["source"].to_list() == ["website", "wikipedia", "total"]
    wiki = out.row(1, named=True)
    assert wiki == {
        "source": "wikipedia",
        "sentences": 2,
        "words": 5,
        "polygons": 2,
        "languages": 1,
        "regions": 2,
        "landuse": 0.5,
    }
    total = out.row(2, named=True)
    assert (total["sentences"], total["words"], total["polygons"], total["languages"]) == (4, 8, 3, 2)
    assert total["landuse"] == 0.75


def test_by_source_order_follows_size():
    data = _data().with_columns(source=pl.Series(["a", "b", "b", "b"]))
    assert stats.by_source(data)["source"].to_list() == ["b", "a", "total"]


def test_top_languages():
    out = stats.top_languages(_data(), 1)
    assert out.rows() == [("fra_Latn", 3)]


def test_by_continent():
    out = stats.by_continent(_data())
    assert out.rows() == [("europe", 3, 3), ("africa", 1, 1)]


def test_points():
    out = stats.points(_data()).sort("lat")
    assert out.rows() == [(1.0, 3.0, 2), (2.0, 4.0, 2)]


def test_unlabelled():
    u = pl.LazyFrame({"source": ["w", "w", "x"], "status": ["failed", "unsplit", "failed"]})
    out = stats.unlabelled(u)
    assert out.rows() == [("w", 1, 1), ("x", 1, 0), ("total", 2, 1)]


def test_ties_break_by_name():
    data = _data().with_columns(
        language=pl.Series(["b", "b", "a", "a"]), continent=pl.Series(["y", "y", "x", "x"])
    )
    assert stats.top_languages(data, 2)["language"].to_list() == ["a", "b"]
    assert stats.by_continent(data)["continent"].to_list() == ["x", "y"]


def test_polygon_areas_quantiles_and_total():
    polygons = pl.LazyFrame({"area_m2": [float(x) for x in range(1, 101)]})
    out = stats.polygon_areas(polygons)
    assert out.columns == ["p10", "p25", "p50", "p75", "p90"]
    assert out.row(0) == (10.0, 25.0, 50.0, 75.0, 90.0)


def _scenes():
    return pl.LazyFrame(
        {
            "text": ["a b", "c d e", "f"],
            "lulc_label": ["Lake / Pond", "Tundra", "Lake / Pond"],
            "n_photos": [3, 10, 2],
            "countries": [["France", "Spain"], ["Norway"], ["France"]],
            "spread_km_p90": [5.0, 100.0, 2_000.0],
        }
    )


def test_scene_totals():
    assert stats.scene_totals(_scenes()) == {
        "scenes": 3,
        "photos": 15,
        "median_photos": 3,
        "words": 6,
        "countries": 3,
    }


def test_scene_labels_by_photos():
    assert stats.scene_labels(_scenes(), 5).rows() == [("Tundra", 1, 10), ("Lake / Pond", 2, 5)]
    assert stats.scene_labels(_scenes(), 1).rows() == [("Tundra", 1, 10)]


def test_scene_labels_ties_by_name():
    tied = _scenes().with_columns(
        n_photos=pl.Series([1, 2, 1]), lulc_label=pl.Series(["Tundra", "Lake", "Tundra"])
    )
    assert stats.scene_labels(tied, 5)["lulc_label"].to_list() == ["Lake", "Tundra"]


def test_scene_platforms():
    photos = pl.LazyFrame({"platform": ["c", "z", "z", "b"], "url": ["u", None, "v", None]})
    assert stats.scene_platforms(photos).rows() == [("z", 2, 0.5), ("b", 1, 0.0), ("c", 1, 1.0)]


def test_scene_locality_is_inclusive():
    assert stats.scene_locality(_scenes()).rows(named=True) == [{"km10": 1, "km100": 2, "km1000": 2}]
