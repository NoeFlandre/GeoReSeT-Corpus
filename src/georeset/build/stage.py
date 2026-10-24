"""Download each (source, region), convert it to staged tables, delete the download."""

import json
import shutil
import tempfile
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from georeset import hub
from georeset.sources.base import Source, Staged

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
    tmp = out.with_name(out.name + ".tmp")  # pragma: no mutate (private scratch name)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for table in TABLES:
        getattr(staged, table).write_parquet(
            tmp / f"{table}.parquet", compression="zstd"
        )  # pragma: no mutate (speed/size only)
    tmp.rename(out)


def run_task(task: Task, work: Path, transforms: threading.Semaphore) -> Path:
    """Download freely, but convert under ``transforms`` to bound memory."""
    out = stage_dir(work, task.source.name, task.stem)
    if out.exists():
        return out
    wanted = {k: p for k, p in task.source.files(task.stem).items() if p in task.available}
    (work / "tmp").mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    with tempfile.TemporaryDirectory(dir=work / "tmp") as tmp:
        local = {k: hub.download(task.source.repo, p, task.rev, Path(tmp)) for k, p in wanted.items()}
        write_staged(task.source.transform(local), out)
    return out


def tasks(sources: Iterable[Source], stems: Iterable[str]) -> list[Task]:
    out = []
    for source in sources:
        rev = hub.revision(source.repo)
        available = hub.list_files(source.repo, rev)
        out += [Task(source, stem, rev, available) for stem in stems]
    return out


def record_revisions(todo: list[Task], work: Path) -> None:
    """Pin the upstream commits used, for the dataset card."""
    revisions = {task.source.repo: task.rev for task in todo}
    work.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    (work / "revisions.json").write_text(json.dumps(revisions, indent=2))


def stage(todo: list[Task], work: Path, downloads: int, transforms: int) -> list[Path]:
    record_revisions(todo, work)
    limit = threading.Semaphore(transforms)
    with ThreadPoolExecutor(downloads) as pool:  # pragma: no mutate (speed/size only)
        return list(pool.map(lambda t: run_task(t, work, limit), todo))
