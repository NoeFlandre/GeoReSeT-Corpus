"""Glue between the work directory and the pure transforms: meta, rows, shards."""

import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from georeset.build.assemble import (
    canonical_docs,
    doc_meta,
    pick_polygons,
    region_priority,
    to_rows,
)
from georeset.build.write import write_shards

_STAGED = r"stage/([^/]+)/([^/]+)/[^/]+\.parquet$"
ROWS_PER_SHARD = 1_000_000
POLYGONS_PER_SHARD = 250_000


@dataclass(frozen=True)
class Layout:
    work: Path
    out: Path

    @property
    def stage(self) -> Path:
        return self.work / "stage"

    @property
    def meta(self) -> Path:
        return self.work / "meta"

    @property
    def rows(self) -> Path:
        return self.work / "rows"


def _scan_staged(layout: Layout, table: str) -> pl.LazyFrame:
    """All staged ``table`` files, with ``source`` and ``stem`` taken from their path."""
    path = pl.col("path")
    return pl.scan_parquet(
        layout.stage / "*" / "*" / f"{table}.parquet", include_file_paths="path"
    ).with_columns(source=path.str.extract(_STAGED, 1), stem=path.str.extract(_STAGED, 2))


def build_meta(layout: Layout) -> None:
    """Pick one row per polygon and one staged copy per document (once)."""
    if (layout.meta / "docs").exists():
        return
    layout.meta.mkdir(parents=True, exist_ok=True)
    candidates = _scan_staged(layout, "polygons").drop("path")
    priority = region_priority(candidates.group_by("stem").agg(n=pl.len())).collect().lazy()
    polygons = pick_polygons(candidates, priority).sort("region", "polygon_id")
    polygons.sink_parquet(layout.meta / "polygons.parquet", compression="zstd")
    attributes = pl.scan_parquet(layout.meta / "polygons.parquet").drop("geometry", "osm_type", "osm_id")
    meta = doc_meta(_scan_staged(layout, "links").select("doc_key", "polygon_id"), attributes)
    presence = _scan_staged(layout, "sentences").select("doc_key", "stem").unique()
    docs = canonical_docs(presence, priority).join(meta, on="doc_key").collect(engine="streaming")
    tmp = layout.meta / "docs.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for (stem,), part in docs.partition_by("stem", as_dict=True).items():
        part.drop("stem").write_parquet(tmp / f"{stem}.parquet")
    tmp.rename(layout.meta / "docs")


def build_rows(layout: Layout, staged: Path) -> None:
    """Final rows of one staged (source, region), partitioned by output source and language.

    The staged region is deleted once its rows are written, to bound disk use.
    """
    source, stem = staged.parent.name, staged.name
    done = layout.rows / "_done" / f"{source}__{stem}"
    docs = layout.meta / "docs" / f"{stem}.parquet"
    if done.exists() or not docs.exists():
        return
    clean, unlabelled = to_rows(pl.read_parquet(staged / "sentences.parquet"), pl.read_parquet(docs))
    name = f"{source}__{stem}.parquet"
    for (out_source, language), part in clean.partition_by("source", "language", as_dict=True).items():
        target = layout.rows / "data" / str(out_source) / str(language)
        target.mkdir(parents=True, exist_ok=True)
        part.write_parquet(target / name)
    (layout.rows / "unlabelled").mkdir(parents=True, exist_ok=True)
    unlabelled.write_parquet(layout.rows / "unlabelled" / name)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    shutil.rmtree(staged)


def _shard(rows: Path, out: Path) -> None:
    """Sort one partition by region then document and write it; rows are deleted after."""
    df = pl.read_parquet(rows / "*.parquet").sort("region", "doc_key", "position")
    write_shards(df.drop("doc_key", "position"), out, ROWS_PER_SHARD)
    shutil.rmtree(rows)


def shard_jobs(layout: Layout) -> list[tuple[Path, Path]]:
    """(rows directory, output directory) for every language partition and unlabelled."""
    todo = [*sorted(layout.rows.glob("data/*/*")), layout.rows / "unlabelled"]
    return [(d, layout.out / d.relative_to(layout.rows)) for d in todo if d.exists()]


def build_shards(layout: Layout, workers: int) -> None:
    run_parallel([lambda r=r, o=o: _shard(r, o) for r, o in shard_jobs(layout)], workers)
    polygons = pl.read_parquet(layout.meta / "polygons.parquet")
    write_shards(
        polygons,
        layout.out / "polygons",
        POLYGONS_PER_SHARD,
        50_000,
        geometry="geometry",
    )


def run_parallel(jobs: Iterable[Callable[[], None]], workers: int) -> None:
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(job) for job in jobs]:
            future.result()
