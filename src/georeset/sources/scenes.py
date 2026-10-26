"""geo-rag scenes: Gemma-4 descriptions of clusters of geotagged photos, tied to every photo's location.

The upstream pipeline ([geo-rag](https://github.com/ananthu-aniraj/geo-rag)) clusters 7.2M photos by
look (TIPSv2 embeddings, k-means) and describes each cluster once. A description therefore belongs to
all the photos of its cluster, which can be far apart: ``spread`` measures how far.
"""

import polars as pl

RAW_REPO = "NoeFlandre/geo-rag-scenes-raw"
PHOTOS_FILE = "geo_space_cleaned.parquet"
CLUSTERS_FILE = "geo_space_google_tipsv2-b14_cls_float16_clustered_k_40000.parquet"
CODE_URL = "https://github.com/ananthu-aniraj/geo-rag"
CODE_REV = "4252e043b3e8f1afbbc1a6cfb55e3577bcb4ee04"

FAILED_TEXT = "Inference failed or returned empty response."
FAILED_LABEL = "Error Labeling"
# Labels outside geo-rag's vocabulary, each on a handful of clusters, mapped to the nearest one.
LABEL_FIXES = {
    "Bare soil / Badlands": "Barren soil / Badlands",
    "Bare rock / Gravel plains": "Rocky desert / Gravel plains",
    "Riparian vegetation / Other natural land cover": "Other natural land cover",
    "Riparian zone / Transportation corridor interface": "Other man-made surface",
    "Coastal settlement": "Other man-made surface",
    "Coastal settlement / Harbor": "Other man-made surface",
}
# Photos whose license forbids reuse keep their id and location, not their URL.
CLOSED_LICENSES = ("All Rights Reserved", "Do not reuse")
UNKNOWN = "Unknown"
EARTH_RADIUS_KM = 6371.0088

SCENE_SCHEMA = pl.Schema(
    {
        "scene_id": pl.Int32,
        "text": pl.String,
        "visual_description": pl.String,
        "lulc_label": pl.String,
        "parent_scene_id": pl.Int32,
        "parent_lulc_label": pl.String,
    }
)
PHOTO_SCHEMA = pl.Schema(
    {
        "scene_id": pl.Int32,
        "platform": pl.String,
        "photo_id": pl.String,
        "url": pl.String,
        "photo_license": pl.String,
        "captured_at": pl.String,
        "season": pl.String,
        "time_of_day": pl.String,
        "koppen": pl.String,
        "country": pl.String,
        "continent": pl.String,
        "h3_cell": pl.String,
        "lat": pl.Float64,
        "lon": pl.Float64,
    }
)


def scenes(clusters: pl.LazyFrame) -> pl.LazyFrame:
    """One row per described cluster; failed descriptions dropped, labels mapped to the vocabulary.

    Every column is constant within a cluster upstream, so the first row of each cluster stands for it.
    """
    first = clusters.group_by("cluster_id").agg(
        pl.col(
            "cluster_description",
            "visual_description",
            "cluster_label",
            "parent_cluster_id",
            "parent_cluster_label",
        ).first()
    )
    labels = pl.col("cluster_label", "parent_cluster_label").replace(LABEL_FIXES)
    failed = (pl.col("cluster_description") == FAILED_TEXT) | (pl.col("cluster_label") == FAILED_LABEL)
    return (
        first.filter(~failed)
        .with_columns(labels, pl.col("visual_description").replace("", None))
        .select(
            scene_id="cluster_id",
            text="cluster_description",
            visual_description="visual_description",
            lulc_label="cluster_label",
            parent_scene_id="parent_cluster_id",
            parent_lulc_label="parent_cluster_label",
        )
        .cast(SCENE_SCHEMA)
    )


def _known(column: str) -> pl.Expr:
    return pl.col(column).replace(UNKNOWN, None)


def continent(column: pl.Expr) -> pl.Expr:
    """geo-rag's ``North America`` to the corpus' ``north-america``; ocean / unknown to null."""
    return (
        pl.when(column.str.contains("/"))
        .then(None)
        .otherwise(column.str.to_lowercase().str.replace_all(" ", "-"))
    )


def url(platform: pl.Expr, image_url: pl.Expr, photo_id: pl.Expr, license_: pl.Expr) -> pl.Expr:
    """Public link to the photo, or null when its license forbids reuse or it is not online.

    Mapillary keys become viewer links; local files (Google Landmarks v2, camera traps) and
    ``kartaview://`` keys have no stable public URL.
    """
    closed = license_.is_in(CLOSED_LICENSES) | license_.is_null()
    mapillary = pl.concat_str(pl.lit("https://www.mapillary.com/app/?pKey="), photo_id)
    return (
        pl.when(closed)
        .then(None)
        .when(platform == "mapillary")
        .then(mapillary)
        .when(image_url.str.starts_with("https://"))
        .then(image_url)
        .otherwise(None)
    )


def photos(meta: pl.LazyFrame, clusters: pl.LazyFrame) -> pl.LazyFrame:
    """One row per clustered photo: where, when, under which license, and its scene."""
    license_ = pl.col("License").str.strip_chars().replace("", None)
    keys = clusters.select("photo_key", "cluster_id")
    return (
        meta.join(keys, on="photo_key")
        .with_columns(License=license_)
        .select(
            scene_id="cluster_id",
            platform="Platform",
            photo_id="Photo_ID",
            url=url(pl.col("Platform"), pl.col("Image_URL"), pl.col("Photo_ID"), pl.col("License")),
            photo_license="License",
            captured_at="Captured_At",
            season=_known("Season"),
            time_of_day=_known("Time_Of_Day"),
            koppen="Koppen_Code",
            country="country",
            continent=continent(pl.col("continent")),
            h3_cell="H3_Cell",
            lat="Latitude",
            lon="Longitude",
        )
        .cast(PHOTO_SCHEMA)
    )


def haversine_km(lat: pl.Expr, lon: pl.Expr, lat0: pl.Expr, lon0: pl.Expr) -> pl.Expr:
    half = ((lat - lat0).radians() / 2).sin() ** 2 + lat.radians().cos() * lat0.radians().cos() * (
        (lon - lon0).radians() / 2
    ).sin() ** 2
    return 2 * EARTH_RADIUS_KM * half.sqrt().arcsin()


def spread(photos: pl.LazyFrame) -> pl.LazyFrame:
    """Per scene: photo count, median location, distance quantiles to it, countries by frequency."""
    centre = photos.group_by("scene_id").agg(
        n_photos=pl.len().cast(pl.Int32), lat=pl.col("lat").median(), lon=pl.col("lon").median()
    )
    distance = haversine_km(pl.col("lat"), pl.col("lon"), pl.col("lat0"), pl.col("lon0"))
    distances = (
        photos.select("scene_id", "lat", "lon")
        .join(centre.select("scene_id", lat0="lat", lon0="lon"), on="scene_id")
        .group_by("scene_id")
        .agg(
            spread_km_p50=distance.quantile(0.5, "linear"),
            spread_km_p90=distance.quantile(0.9, "linear"),
        )
    )
    countries = (
        photos.group_by("scene_id", "country")
        .agg(n=pl.len())
        .sort("scene_id", "n", "country", descending=[False, True, False])
        .group_by("scene_id", maintain_order=True)
        .agg(countries=pl.col("country"))
    )
    return centre.join(distances, on="scene_id").join(countries, on="scene_id")


def with_spread(scene_rows: pl.LazyFrame, photo_rows: pl.LazyFrame) -> pl.LazyFrame:
    """Scenes that kept at least one photo, with their spread, sorted by id."""
    return scene_rows.join(spread(photo_rows), on="scene_id").sort("scene_id")
