"""Write frames as ``train-XXXXX-of-YYYYY.parquet`` shards, atomically per directory."""

import json
import math
import shutil
from collections.abc import Iterable
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


class _Shards:
    """Fills ``count`` shards of ``size`` rows in order, opening each file on demand."""

    def __init__(self, folder: Path, count: int, size: int, row_group_size: int, geometry: str | None):
        self.folder, self.count, self.size = folder, count, size
        self.row_group_size, self.geometry = row_group_size, geometry
        self.index, self.filled, self.writer = 0, 0, None

    def _open(self, table: pa.Table) -> pq.ParquetWriter:
        name = f"train-{self.index:05d}-of-{self.count:05d}.parquet"
        return pq.ParquetWriter(
            self.folder / name, table.schema, compression="zstd"
        )  # pragma: no mutate (case-insensitive)

    def write(self, df: pl.DataFrame) -> None:
        while df.height:
            part = df.head(self.size - self.filled)
            table = _table(part, self.geometry)
            self.writer = self.writer or self._open(table)
            self.writer.write_table(table, row_group_size=self.row_group_size)
            self.filled += part.height
            df = df.slice(part.height)
            if self.filled == self.size:
                self.close()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.index, self.filled, self.writer = self.index + 1, 0, None


def write_stream(
    frames: Iterable[pl.DataFrame],
    total: int,
    out: Path,
    rows_per_shard: int,
    row_group_size: int = 100_000,  # pragma: no mutate (speed/size only)
    geometry: str | None = None,
) -> list[Path]:
    """Replace ``out`` with ``total`` rows streamed from ``frames``; returns the shard paths."""
    if total == 0:
        return []
    tmp = out.with_name(out.name + ".tmp")  # pragma: no mutate (private scratch name)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shards = _Shards(tmp, math.ceil(total / rows_per_shard), rows_per_shard, row_group_size, geometry)
    for frame in frames:
        shards.write(frame)
    shards.close()
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return sorted(out.iterdir())


def write_shards(
    df: pl.DataFrame,
    out: Path,
    rows_per_shard: int,
    row_group_size: int = 100_000,  # pragma: no mutate (speed/size only)
    geometry: str | None = None,
) -> list[Path]:
    """Replace ``out`` with ``df`` split into shards; returns the shard paths."""
    return write_stream([df], df.height, out, rows_per_shard, row_group_size, geometry)
