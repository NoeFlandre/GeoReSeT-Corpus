"""Partitioning the round's join by the first hex digit of the text sha256.

The filter's generations (about 80 GB) do not fit in memory next to the corpus rows, so both sides are
written into 16 buckets by the same rule, and each bucket is joined on its own. Rows join on their label
repository (the group) as well as the text: the same text can be generated in two repositories.
"""

from pathlib import Path
from typing import Any

import polars as pl

from georeset.label import backfill
from georeset.label.text import sha256_text

BUCKETS = tuple(f"{digit:x}" for digit in range(16))
# The label repository (group) that labelled each corpus source.
GROUP_OF_SOURCE = {
    "osm_description": "description",
    "website": "website",
    "wikipedia": "wikipedia",
    "wikivoyage": "wikipedia",
}
CORPUS_SCHEMA: dict[str, Any] = {
    "id": pl.String,
    "text_sha256": pl.String,
    "landuse": pl.Boolean,
    "group": pl.String,
}
GENERATION_SCHEMA: dict[str, Any] = {
    "group": pl.String,
    "text_sha256": pl.String,
    "raw_output": pl.String,
    "finish_reason": pl.String,
    "truncated": pl.Boolean,
}


def corpus_file_rows(path: Path) -> pl.DataFrame:
    """A corpus data file as (id, text sha256, landuse, group); the text itself is not kept."""
    rows = pl.read_parquet(path, columns=["id", "text", "landuse", "source"])
    hashes = [sha256_text(text) for text in rows["text"].to_list()]
    groups = [GROUP_OF_SOURCE[source] for source in rows["source"].to_list()]
    return pl.DataFrame(
        {
            "id": rows["id"],
            "text_sha256": pl.Series(hashes, dtype=pl.String),
            "landuse": rows["landuse"],
            "group": pl.Series(groups, dtype=pl.String),
        }
    )


def generation_file_rows(path: Path, group: str) -> pl.DataFrame:
    """A generations file of one label repository, with the columns the round keeps."""
    columns = [name for name in GENERATION_SCHEMA if name != "group"]
    return pl.read_parquet(path, columns=columns).with_columns(group=pl.lit(group))


def write_buckets(frame: pl.DataFrame, out: Path, name: str) -> None:
    """Write each bucket's rows of ``frame`` to ``out/<bucket>/<name>.parquet``; empty buckets get no file."""
    for bucket in BUCKETS:
        part = frame.filter(pl.col("text_sha256").str.starts_with(bucket))
        if part.height:
            path = out / bucket / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(path)


def _read_bucket(folder: Path, schema: dict[str, Any]) -> pl.DataFrame:
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        return pl.DataFrame(schema=schema)
    return pl.concat([pl.read_parquet(path) for path in files])


def join_bucket(
    bucket: str, corpus_dir: Path, generations_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """One bucket of the round: its rows, its generations, and the agreement of the parser with the labels."""
    corpus = _read_bucket(corpus_dir / bucket, CORPUS_SCHEMA)
    generations = _read_bucket(generations_dir / bucket, GENERATION_SCHEMA)
    return (
        backfill.round_rows(corpus, generations),
        backfill.raw_rows(corpus, generations),
        backfill.agreement(corpus, generations),
    )
