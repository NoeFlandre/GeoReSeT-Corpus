import polars as pl
import shapely

from georeset.normalize.osm import (
    geojson_to_wkb,
    has_letters,
    polygon_id,
    primary_tag,
    tags_list_to_json,
)


def _eval(expr: pl.Expr, **columns) -> list:
    return pl.DataFrame(columns).select(expr.alias("out"))["out"].to_list()


def test_polygon_id():
    assert _eval(polygon_id(pl.col("t"), pl.col("i")), t=["way", "relation"], i=[1, 22]) == [
        "way/1",
        "relation/22",
    ]


def test_primary_tag_follows_priority():
    tags = [
        '{"building":"yes","amenity":"school","landuse":"education"}',
        '{"building":"yes"}',
    ]
    assert _eval(primary_tag(pl.col("j")), j=tags) == [
        "landuse=education",
        "building=yes",
    ]


def test_primary_tag_tolerates_spaces_and_missing():
    assert _eval(primary_tag(pl.col("j")), j=['{"natural": "wood"}', '{"name":"x"}', None]) == [
        "natural=wood",
        None,
        None,
    ]


def test_primary_tag_matches_whole_key():
    assert _eval(primary_tag(pl.col("j")), j=['{"disused:shop":"bakery","shop":"bread"}']) == ["shop=bread"]


def test_tags_list_to_json_feeds_primary_tag():
    tags = [
        [{"key": "name", "value": "Refuge"}, {"key": "tourism", "value": "alpine_hut"}],
        [],
    ]
    out = _eval(primary_tag(tags_list_to_json(pl.col("t"))), t=tags)
    assert out == ["tourism=alpine_hut", None]


def test_tags_list_to_json_format():
    tags = [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]
    assert _eval(tags_list_to_json(pl.col("t")), t=tags) == ['{"a":"1","b":"2"}']


def test_geojson_to_wkb_roundtrip_and_invalid():
    square = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
    out = geojson_to_wkb(pl.Series("g", [square, "not json"]))
    assert out.dtype == pl.Binary
    assert out.name == "g"
    assert shapely.from_wkb(out[0]).area == 1.0
    assert out[1] is None


def test_has_letters():
    texts = ["-", "7", "ab", "a1", "Été", "東京", "• x"]
    assert _eval(has_letters(pl.col("t")), t=texts) == [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
    ]


def test_has_letters_custom_minimum():
    assert _eval(has_letters(pl.col("t"), 1), t=["a", "1"]) == [True, False]


def test_geojson_to_wkb_keeps_binary_type_when_all_invalid():
    out = geojson_to_wkb(pl.Series("g", ["bad", None]))
    assert out.dtype == pl.Binary
    assert out.to_list() == [None, None]
