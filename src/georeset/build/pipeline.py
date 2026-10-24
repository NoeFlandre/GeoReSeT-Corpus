"""Glue between the work directory and the pure transforms: meta, rows, shards."""

import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from georeset.build.assemble import (
    canonical_docs,
    doc_meta,
    pick_polygons,
    region_priority,
    to_rows,
)
from georeset.build.write import write_stream

_STAGED = r"stage/([^/]+)/([^/]+)/[^/]+\.parquet$"
ROWS_PER_SHARD = 1_000_000
SORT_CHUNK_ROWS = 3_000_000
POLYGONS_PER_SHARD = 250_000
POLYGON_ROW_GROUP = 50_000  # batch size, hence row group size, of the polygon shards


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
    ).with_columns(source=path.str.extract(_STAGED), stem=path.str.extract(_STAGED, 2))


def build_meta(layout: Layout) -> None:
    """Pick one row per polygon and one staged copy per document (once)."""
    if (layout.meta / "docs").exists():
        return
    layout.meta.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    candidates = _scan_staged(layout, "polygons").drop("path")
    priority = region_priority(candidates.group_by("stem").agg(n=pl.len())).collect().lazy()
    polygons = pick_polygons(candidates, priority).sort("region", "polygon_id")
    polygons.sink_parquet(
        layout.meta / "polygons.parquet", compression="zstd"
    )  # pragma: no mutate (speed/size only)
    attributes = pl.scan_parquet(layout.meta / "polygons.parquet")
    meta = doc_meta(_scan_staged(layout, "links").select("doc_key", "polygon_id"), attributes)
    presence = _scan_staged(layout, "sentences").select("doc_key", "stem").unique()
    docs = canonical_docs(presence, priority).join(meta, on="doc_key").collect(engine="streaming")
    tmp = layout.meta / "docs.tmp"  # pragma: no mutate (private scratch name)
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
        target.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
        part.write_parquet(target / name)
    (layout.rows / "unlabelled").mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    unlabelled.write_parquet(layout.rows / "unlabelled" / name)
    done.parent.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    done.touch()
    shutil.rmtree(staged)


def region_chunks(counts: pl.DataFrame, budget: int) -> list[list[str]]:
    """Group consecutive sorted regions into chunks of at most ``budget`` rows (or one region)."""
    chunks: list[list[str]] = []
    size = budget
    for region, n in counts.sort("region").iter_rows():
        if size + n > budget:
            chunks.append([])
            size = 0
        chunks[-1].append(region)
        size += n
    return chunks


def _shard(rows: Path, out: Path) -> None:
    """Sort one partition by region then document, chunk by chunk to bound memory; delete rows after."""
    scan = pl.scan_parquet(rows / "*.parquet")
    counts = scan.group_by("region").agg(n=pl.len()).collect()
    frames = (
        scan.filter(pl.col("region").is_in(chunk))
        .collect()
        .sort("region", "doc_key", "position")
        .drop("doc_key", "position")
        for chunk in region_chunks(counts, SORT_CHUNK_ROWS)
    )
    write_stream(frames, int(counts["n"].sum()), out, ROWS_PER_SHARD)
    shutil.rmtree(rows)


def shard_jobs(layout: Layout) -> list[tuple[Path, Path]]:
    """(rows directory, output directory) for every language partition and unlabelled."""
    todo = [*sorted(layout.rows.glob("data/*/*")), layout.rows / "unlabelled"]
    return [(d, layout.out / d.relative_to(layout.rows)) for d in todo if d.exists()]


def build_shards(layout: Layout, workers: int) -> None:
    run_parallel(
        [lambda r=r, o=o: _shard(r, o) for r, o in shard_jobs(layout)], workers
    )  # pragma: no mutate (speed/size only)
    source = pq.ParquetFile(layout.meta / "polygons.parquet")
    batches = (pl.DataFrame(b) for b in source.iter_batches(batch_size=POLYGON_ROW_GROUP))
    out = layout.out / "polygons"
    total = source.metadata.num_rows
    write_stream(batches, total, out, POLYGONS_PER_SHARD, geometry="geometry")


def run_parallel(jobs: Iterable[Callable[[], None]], workers: int) -> None:
    with ThreadPoolExecutor(workers) as pool:  # pragma: no mutate (speed/size only)
        for future in [pool.submit(job) for job in jobs]:
            future.result()
