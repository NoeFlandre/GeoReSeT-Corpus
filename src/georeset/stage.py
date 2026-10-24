"""Download each (source, region), convert it to staged tables, delete the download."""

import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from georeset import hub
from georeset.sources import Source, Staged

TABLES = ("polygons", "links", "sentences")


@dataclass(frozen=True)
class Task:
    source: Source
    stem: str
    rev: str
    available: frozenset[str]


def stage_dir(work: Path, source: str, stem: str) -> Path:
    return work / "stage" / source / stem


def write_staged(staged: Staged, out: Path) -> None:
    """Write the three tables atomically: a crash never leaves a half-staged region."""
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for table in TABLES:
        getattr(staged, table).write_parquet(
            tmp / f"{table}.parquet", compression="zstd"
        )
    tmp.rename(out)


def run_task(task: Task, work: Path) -> Path:
    out = stage_dir(work, task.source.name, task.stem)
    if out.exists():
        return out
    wanted = {
        k: p for k, p in task.source.files(task.stem).items() if p in task.available
    }
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work / "tmp") as tmp:
        local = {
            k: hub.download(task.source.repo, p, task.rev, Path(tmp))
            for k, p in wanted.items()
        }
        write_staged(task.source.transform(local), out)
    return out


def tasks(sources: Iterable[Source], stems: Iterable[str]) -> list[Task]:
    out = []
    for source in sources:
        rev = hub.revision(source.repo)
        available = hub.list_files(source.repo, rev)
        out += [Task(source, stem, rev, available) for stem in stems]
    return out


def stage(todo: list[Task], work: Path, workers: int) -> list[Path]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda t: run_task(t, work), todo))
