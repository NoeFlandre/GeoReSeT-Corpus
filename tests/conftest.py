from pathlib import Path

import polars as pl
import pytest

SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'


@pytest.fixture
def write(tmp_path: Path):
    """Write a dict of columns to ``tmp_path/name.parquet`` and return the path."""

    def _write(name: str, columns: dict, schema: dict | None = None) -> Path:
        path = tmp_path / f"{name}.parquet"
        pl.DataFrame(columns, schema=schema).write_parquet(path)
        return path

    return _write
