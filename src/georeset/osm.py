"""Vectorized helpers shared by all sources: ids, primary tags, geometry, text filter."""

import polars as pl
import shapely

# First key found wins; ordered from land-use specific to generic.
PRIMARY_KEYS = (
    "landuse",
    "natural",
    "leisure",
    "amenity",
    "tourism",
    "historic",
    "shop",
    "office",
    "craft",
    "man_made",
    "aeroway",
    "railway",
    "highway",
    "waterway",
    "water",
    "military",
    "power",
    "sport",
    "healthcare",
    "public_transport",
    "emergency",
    "place",
    "boundary",
    "building",
    "barrier",
)


def polygon_id(osm_type: pl.Expr, osm_id: pl.Expr) -> pl.Expr:
    """``way/123`` style id, stable across regions and sources."""
    return pl.concat_str(osm_type, pl.lit("/"), osm_id.cast(pl.String))


def tags_list_to_json(tags: pl.Expr) -> pl.Expr:
    """Render a ``list<struct<key, value>>`` tag column as a JSON-like object string."""
    pair = pl.format(
        '"{}":"{}"',
        pl.element().struct.field("key"),
        pl.element().struct.field("value"),
    )
    return pl.concat_str(pl.lit("{"), tags.list.eval(pair).list.join(","), pl.lit("}"))


def primary_tag(tags_json: pl.Expr) -> pl.Expr:
    """``key=value`` of the first ``PRIMARY_KEYS`` key present in a JSON tag object."""
    return pl.coalesce(
        pl.concat_str(
            pl.lit(f"{key}="), tags_json.str.extract(f'"{key}":\\s*"([^"]*)"')
        )
        for key in PRIMARY_KEYS
    )


def geojson_to_wkb(geojson: pl.Series) -> pl.Series:
    """Convert GeoJSON strings to WKB, invalid input becomes null."""
    geoms = shapely.from_geojson(geojson.to_numpy(), on_invalid="ignore")
    return pl.Series(geojson.name, shapely.to_wkb(geoms), dtype=pl.Binary)


def has_letters(text: pl.Expr, minimum: int = 2) -> pl.Expr:
    """True when ``text`` holds at least ``minimum`` Unicode letters."""
    return text.str.count_matches(r"\p{L}") >= minimum
