"""The migration: a corpus data file without its inline ``landuse``; the labels live in the round's file."""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

LABEL = "landuse"


def strip_label(src: Path, dst: Path) -> int:
    """Write ``src`` to ``dst`` without the label column; returns the number of rows written."""
    frame = pl.read_parquet(src).drop(LABEL)
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dst)
    return frame.height


def rows(paths: list[Path]) -> int:
    """Rows in the parquet files, read from their footers."""
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def inline_labels(paths: list[Path]) -> int:
    """How many of the files still carry the inline label column."""
    return sum(LABEL in pq.read_schema(path).names for path in paths)


def check_migrated(data_rows: int, decision_rows: int, inline: int) -> None:
    """Raise unless the migration is complete: no inline label left, one decision per data row."""
    if inline:
        raise ValueError(f"{inline} data files still have the inline {LABEL} column")
    if decision_rows != data_rows:
        raise ValueError(f"{decision_rows} decisions for {data_rows} data rows")
