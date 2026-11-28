"""The relevant flag of a corpus row.

A release before v2.0 carries ``landuse`` inline. From v2.0 the labels live in the current round's sidecar,
so the flag is the round's ``yes`` decision, joined on id.
"""

from pathlib import Path

import polars as pl

from georeset.label import prompt, rounds

CURRENT = rounds.name_for(rounds.SLUG, prompt.PROMPT_SHA256)


def landuse(data: pl.LazyFrame, out: Path) -> pl.LazyFrame:
    """The rows with a boolean ``landuse``, read from the folder ``out`` when it has no column."""
    if "landuse" in data.collect_schema().names():
        return data
    decisions = pl.scan_parquet(str(out / "labels" / CURRENT / "decisions-*.parquet")).select(
        "id", landuse=pl.col("decision") == "yes"
    )
    return data.join(decisions, on="id", how="left")
