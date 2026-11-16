"""The translation's steps as they run on Grid'5000: plan, translate (GPU jobs), assemble, clean up.

Every step reads the release from the Hub and the working files from the bucket, keeps its own files in
node-local scratch space, and deletes them as soon as they are uploaded.
"""

import json
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import polars as pl

from georeset import hub, workqueue
from georeset.build import fuse
from georeset.translate import assemble, models, plan, route, worker
from georeset.translate.store import Bucket

REPO = "NoeFlandre/GeoReSeT-Corpus"
RELEASE = "v1.0"  # the release the translations are added to
MEMORY_GB = 16  # NLLB 3.3B in float16 and its batches
MODEL_DIR = "model"


def release(repo: str = REPO, tag: str = RELEASE) -> str:
    """The commit of the release, pinned for every step."""
    return hub.revision_of(repo, tag)


def planned(bucket: Bucket) -> set[str]:
    return {Path(p).stem for p in bucket.files("plan/")}


def plan_unit(unit: str, rev: str, bucket: Bucket, scratch: Path, identify: plan.Identify) -> None:
    """Plan one unit unless already done; its summary is uploaded last, as the mark that it is done."""
    if unit in planned(bucket):
        return
    folder = scratch / unit
    hub.snapshot(REPO, rev, folder / "in", [plan.pattern(unit)])
    unit_plan = plan.Plan()
    for path in sorted((folder / "in").rglob("*.parquet")):
        unit_plan.add(plan.rows(path), identify)
        path.unlink()
    *files, summary = plan.write(unit, unit_plan, folder / "out")
    bucket.put(files)
    bucket.put([summary])
    shutil.rmtree(folder)
    print(f"planned {unit}", flush=True)


_IDENTIFY: Callable[[list[str]], tuple[list[str], list[float]]] | None = None


def _init_identify() -> None:  # pragma: no cover (loads GlotLID in a worker process)
    global _IDENTIFY
    _IDENTIFY = models.GlotLid().predict


def _plan_in_process(unit: str, rev: str, scratch: Path) -> None:  # pragma: no cover (worker process)
    assert _IDENTIFY is not None
    plan_unit(unit, rev, Bucket(), scratch, _IDENTIFY)


def plan_all(scratch: Path, workers: int, only: list[str] | None = None) -> None:  # pragma: no cover
    """Plan every unit of the release (or ``only`` those), ``workers`` at a time (Grid'5000 CPU job)."""
    rev = release()
    bucket = Bucket()
    bucket.create()
    units = only or plan.units(hub.list_files(REPO, rev))
    todo = [u for u in units if u not in planned(bucket)]
    with ProcessPoolExecutor(workers, initializer=_init_identify) as pool:
        list(pool.map(_plan_in_process, todo, [rev] * len(todo), [scratch] * len(todo)))


def pending(bucket: Bucket) -> str:
    """The dispatcher's work items (``g5k-dispatch``): one, the work shards left and the shards done.

    Planning may still add work shards; the result shards only grow, so they measure progress.
    """
    remaining = len(bucket.files("work/"))
    done = len(bucket.files("results/"))
    item = {"key": "nllb", "memory_gb": MEMORY_GB, "remaining": remaining, "fp16_ok": True, "done": done}
    return json.dumps([item])


def fetch_model(bucket: Bucket, scratch: Path) -> Path:
    """Download the converted model into ``scratch``."""
    folder = scratch / MODEL_DIR
    files = bucket.files(f"{MODEL_DIR}/")
    bucket.get([(path, scratch / path) for path in files])
    return folder


def translate(scratch: Path, offset: int, budget_s: float) -> int:  # pragma: no cover (GPU job)
    bucket = Bucket()
    engine = models.Nllb(fetch_model(bucket, scratch))
    lid = models.GlotLid()
    transform = partial(worker.translate_shard, engine, lid.labels)
    return workqueue.Worker(bucket, transform, scratch, offset, budget_s).run()


def assemble_unit(unit: str, out: Path, bucket: Bucket, scratch: Path) -> int:
    """Add the translation columns to every shard of one unit in ``out``; returns the shards done."""
    folder = scratch / unit
    files = [*bucket.files(f"results/{unit}/"), f"routes/{unit}.parquet"]
    bucket.get([(p, folder / p) for p in files])
    translations = assemble.load_translations(sorted((folder / "results").rglob("*.parquet")))
    overrides = pl.read_parquet(folder / "routes" / f"{unit}.parquet")
    shards = sorted(out.glob(plan.pattern(unit)))
    for path in shards:
        assemble.rewrite(path, overrides, translations)
    shutil.rmtree(folder)
    return len(shards)


def summary(bucket: Bucket, scratch: Path, rev: str) -> dict:
    """What the card needs from the plan: distinct texts, model, settings."""
    files = bucket.files("plan/")
    bucket.get([(p, scratch / p) for p in files])
    distinct = sum(json.loads((scratch / p).read_text())["distinct"] for p in files)
    return {
        "distinct": distinct,
        "model": models.NLLB_REPO,
        "revision": models.NLLB_REVISION,
        "threshold": route.THRESHOLD,
        "release": rev,
    }


def assemble_all(out: Path, work: Path, scratch: Path) -> None:  # pragma: no cover (Grid'5000 CPU job)
    """Copy the release into ``out``, add the translation columns, write ``work/translation.json``."""
    rev = release()
    bucket = Bucket()
    if bucket.files("work/"):
        raise RuntimeError("work shards are left: translation is not finished")
    fuse.fetch_corpus(REPO, rev, out, work)
    for unit in plan.units(hub.list_files(REPO, rev)):
        print(f"{unit}: {assemble_unit(unit, out, bucket, scratch)} shards", flush=True)
    (work / "translation.json").write_text(json.dumps(summary(bucket, scratch, rev), indent=2))


def upload_model(folder: Path, bucket: Bucket) -> None:
    """Upload the converted model's files under ``model/``."""
    bucket.create()
    bucket.put([(path, f"{MODEL_DIR}/{path.name}") for path in sorted(folder.iterdir()) if path.is_file()])


NEXT_RELEASE = "v1.1"
PR_TITLE = (
    "Add English translations (NLLB-200 3.3B): text_en, translation, translation_language, translation_flags"
)


def propose(out: Path) -> str:
    """Open a pull request with the translated release, for review; ``polygons`` (unchanged) is left out."""
    shutil.rmtree(out / "polygons", ignore_errors=True)
    return hub.propose(out, REPO, PR_TITLE)


def finalize(number: int, bucket: Bucket) -> None:
    """Merge the reviewed pull request, tag the release, delete the bucket."""
    hub.merge(REPO, number)
    hub.tag(REPO, NEXT_RELEASE)
    bucket.remove()
