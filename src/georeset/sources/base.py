"""Source adapters: one region of one upstream repo to the common staged tables."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from georeset.normalize.languages import Scheme, normalize
from georeset.normalize.osm import has_letters

POLYGON_SCHEMA = pl.Schema(
    {
        "polygon_id": pl.String,
        "name": pl.String,
        "osm_tag": pl.String,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "area_m2": pl.Float64,
        "geometry": pl.Binary,
    }
)
LINK_SCHEMA = pl.Schema({"doc_key": pl.String, "polygon_id": pl.String})
SENTENCE_SCHEMA = pl.Schema(
    {
        "source": pl.String,
        "doc_key": pl.String,
        "position": pl.Int32,
        "text": pl.String,
        "label": pl.String,
        "failure_reason": pl.String,
        "language": pl.String,
        "url": pl.String,
    }
)


@dataclass(frozen=True)
class Staged:
    polygons: pl.DataFrame
    links: pl.DataFrame
    sentences: pl.DataFrame


@dataclass(frozen=True)
class Source:
    """An upstream repo: which files a region needs and how to turn them into ``Staged``."""

    name: str
    repo: str
    files: Callable[[str], dict[str, str]]
    transform: Callable[[dict[str, Path]], Staged]


def finish_sentences(sentences: pl.DataFrame, scheme: Scheme) -> pl.DataFrame:
    """Drop debris (< 2 letters), normalize languages, enforce the common schema."""
    kept = sentences.filter(has_letters(pl.col("text")))
    codes = kept["language"].unique().to_list()
    mapping = {code: normalize(code, scheme) for code in codes}
    return (
        kept.with_columns(pl.col("language").replace_strict(mapping))
        .select(SENTENCE_SCHEMA.names())
        .cast(SENTENCE_SCHEMA)
    )


def attach_text(labels: pl.DataFrame, docs: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Add ``text`` to labels keyed by ``keys`` + ``sentence_index``.

    ``docs`` holds ``keys``, a ``sentences`` list and the ``full_text`` used for
    ``skipped_unsplit`` labels. Lists are exploded, never joined, to keep memory flat.
    """
    exploded = (
        docs.select(
            *keys,
            "sentences",
            sentence_index=pl.int_ranges(pl.col("sentences").list.len()),
        )
        .explode("sentences", "sentence_index")
        .drop_nulls("sentence_index")
    )
    joined = labels.join(exploded, on=[*keys, "sentence_index"], how="left").join(
        docs.select(*keys, "full_text"), on=keys
    )
    skipped = pl.col("label") == "skipped_unsplit"
    return joined.with_columns(text=pl.when(skipped).then("full_text").otherwise("sentences")).drop(
        "sentences", "full_text"
    )


def osm_url(polygon_id: pl.Expr) -> pl.Expr:
    return pl.concat_str(pl.lit("https://www.openstreetmap.org/"), polygon_id)
