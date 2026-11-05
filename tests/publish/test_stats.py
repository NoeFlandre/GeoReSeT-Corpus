import numpy as np
import polars as pl
import pytest

from georeset.publish import geo, stats


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


def test_landuse_sentences_keeps_relevant_distinct_texts():
    data = pl.LazyFrame(
        {
            "id": ["c", "a", "b", "d"],
            "text": ["x", "x", "y", "z"],
            "landuse": [True, True, False, True],
            "language": ["fra_Latn", "eng_Latn", "fra_Latn", "deu_Latn"],
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lon": [5.0, 6.0, 7.0, 8.0],
            "source": ["w", "w", "w", "w"],
        }
    )
    out = stats.landuse_sentences(data).collect()
    assert out.columns == ["id", "text", "language", "lat", "lon"]
    assert out.rows() == [("a", "x", "eng_Latn", 2.0, 6.0), ("d", "z", "deu_Latn", 4.0, 8.0)]


def test_hexagon_sums():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    sums, counts = stats.hexagon_sums(np.array([3, -1, 3, 0]), vectors)
    assert sums.shape == (geo.HEXAGONS, 2)
    assert sums[3].tolist() == [6.0, 8.0]
    assert sums[0].tolist() == [7.0, 8.0]
    assert sums.sum() == 29.0
    assert counts[3] == 2 and counts[0] == 1 and counts.sum() == 3
    assert counts.shape == (geo.HEXAGONS,)


def test_hexagon_sums_of_nothing_inside():
    sums, counts = stats.hexagon_sums(np.array([-1]), np.ones((1, 2)))
    assert sums.sum() == 0 and counts.sum() == 0


def test_pca_recovers_the_main_direction():
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)
    x = np.column_stack([3 * t, -3 * t, rng.normal(scale=0.1, size=500)]) + 5
    fit = stats.pca(x, 2)
    assert fit.components.shape == (2, 3)
    assert np.allclose(np.abs(fit.components[0]), [2**-0.5, 2**-0.5, 0], atol=0.01)
    assert fit.components[0][np.abs(fit.components[0]).argmax()] > 0
    assert fit.explained[0] > 0.99 and fit.explained.sum() <= 1
    assert np.allclose(fit.mean, x.mean(axis=0))
    scores = fit.scores(x)
    assert np.allclose(scores.mean(axis=0), 0)
    assert np.allclose(scores[:, 0] ** 2, ((x - x.mean(axis=0)) @ fit.components[0]) ** 2)


def test_pca_sign_is_fixed_by_the_largest_loading():
    x = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.1]])
    assert stats.pca(x, 1).components[0][0] > 0
    assert stats.pca(-x, 1).components[0][0] > 0


def test_unit_scale_clips_at_percentiles():
    scores = np.column_stack([np.arange(101.0), np.full(101, 4.0)])
    out = stats.unit_scale(scores)
    assert out[0, 0] == 0 and out[100, 0] == 1
    assert out[50, 0] == pytest.approx(0.5)
    assert out[2, 0] == 0 and out[98, 0] == 1
    assert out[3, 0] == pytest.approx(1 / 96)
    assert (out[:, 1] == 0).all()


def test_symmetric_limit():
    assert stats.symmetric_limit(np.arange(-10.0, 91.0)) == pytest.approx(88.0)
    assert stats.symmetric_limit(np.arange(-90.0, 11.0)) == pytest.approx(88.0)


def test_dominant_language_breaks_ties_by_name():
    counts = pl.DataFrame(
        {"hexagon": [1, 1, 2, 2, 2], "language": ["b", "a", "z", "y", "x"], "n": [5, 5, 1, 3, 2]}
    )
    assert stats.dominant_language(counts).sort("hexagon").rows() == [(1, "a"), (2, "y")]


def test_r_squared():
    scores = np.array([[1.0, 1.0], [3.0, 2.0], [10.0, 1.0], [12.0, 2.0]])
    r2 = stats.r_squared(scores, ["a", "a", "b", "b"])
    assert r2[0] == pytest.approx(1 - 4 / 85)
    assert r2[1] == pytest.approx(0)


def test_extremes_and_language_mix():
    scored = pl.DataFrame(
        {"id": ["a", "b", "c", "d"], "language": ["x", "y", "x", "x"], "score": [3.0, 1.0, 3.0, -2.0]}
    )
    high, low = stats.extremes(scored, "score", 2)
    assert high["id"].to_list() == ["a", "c"]
    assert low["id"].to_list() == ["d", "b"]
    assert stats.language_mix(scored).rows() == [("x", 0.75), ("y", 0.25)]


def test_photo_hexagons_pairs_each_scene_with_its_photos_hexagons():
    photos = pl.LazyFrame(
        {
            "scene_id": [1, 1, 1, 2, 2],
            "lat": [48.85, 48.86, -33.9, 48.85, 95.0],
            "lon": [2.35, 2.36, 18.4, 2.35, 0.0],
        }
    )
    paris, cape = geo.hexagons(np.array([2.35, 18.4]), np.array([48.85, -33.9]))
    expected = sorted([(1, paris), (1, cape), (2, paris)])
    assert stats.photo_hexagons(photos).rows() == expected
