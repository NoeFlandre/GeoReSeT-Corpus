"""Write a frame as ``train-XXXXX-of-YYYYY.parquet`` shards, atomically per directory."""

import json
import math
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _geo_metadata(column: str) -> dict[bytes, bytes]:
    geo = {
        "version": "1.1.0",
        "primary_column": column,
        "columns": {column: {"encoding": "WKB", "geometry_types": []}},
    }
    return {b"geo": json.dumps(geo).encode()}


def _table(df: pl.DataFrame, geometry: str | None) -> pa.Table:
    table = df.to_arrow()
    return table.replace_schema_metadata(_geo_metadata(geometry)) if geometry else table


def write_shards(
    df: pl.DataFrame,
    out: Path,
    rows_per_shard: int,
    row_group_size: int = 100_000,
    geometry: str | None = None,
) -> list[Path]:
    """Replace ``out`` with ``df`` split into shards; returns the shard paths."""
    if df.is_empty():
        return []
    count = math.ceil(df.height / rows_per_shard)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for index in range(count):
        part = df.slice(index * rows_per_shard, rows_per_shard)
        pq.write_table(
            _table(part, geometry),
            tmp / f"train-{index:05d}-of-{count:05d}.parquet",
            row_group_size=row_group_size,
            compression="zstd",
        )
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return sorted(out.iterdir())
