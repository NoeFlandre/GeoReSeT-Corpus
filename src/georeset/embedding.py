"""Embed the English translations of the relevant sentences on Grid'5000, as the translation was run.

    plan (CPU job)     the release's relevant ``nllb`` rows, one per distinct ``text_en`` (the smallest
                       ``id`` keeps it), in work shards in the private bucket ``BUCKET``
    work (GPU jobs)    ``georeset.workqueue``: each shard's vectors under ``results/``, one job per free GPU
                       (g5k-dispatch)
    assemble (CPU job) the release, its card context and the vectors as ``embeddings_en``, checked; a pull
                       request on the dataset; ``finalize`` merges it, tags the release, deletes the bucket

The model, instruction and settings are those of ``embeddings`` (``scripts/embed.py``), so that both sets of
vectors live in one space. Vectors are L2-normalised and stored as float16.
"""

import json
import os
import shutil
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from georeset import hub, workqueue
from georeset.build import extras, fuse
from georeset.publish import themes
from georeset.translate.store import Bucket

BUCKET = "NoeFlandre/georeset-embeddings"
REPO = extras.REPO
RELEASE, NEXT_RELEASE = "v1.2", "v1.3"
FOLDER = "embeddings_en"
RELEASE_FOLDERS = (*fuse.CORPUS_FOLDERS, *extras.EXTRAS_FOLDERS)
ROWS_PER_SHARD = 50_000  # a few minutes on one GPU
BATCH = 256
MEMORY_GB = 16
PR_TITLE = "Add embeddings_en: vectors of the English translations of the relevant sentences"

Encode = Callable[[list[str]], np.ndarray]


def selection(data: pl.LazyFrame) -> pl.LazyFrame:
    """Relevant translated sentences, one per distinct English text (the smallest ``id`` keeps it)."""
    rows = data.filter(pl.col("landuse") & (pl.col("translation") == "nllb"))
    english = rows.select("id", pl.col("text_en").alias("text"), "lat", "lon")
    return english.sort("id").unique("text", keep="first", maintain_order=True)


def work_shards(rows: pl.DataFrame, folder: Path, size: int = ROWS_PER_SHARD) -> list[tuple[Path, str]]:
    """``rows`` written in shards of ``size`` rows; ``(local file, path in the bucket)`` pairs."""
    folder.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, start in enumerate(range(0, rows.height, size)):
        path = folder / f"{i:05d}.parquet"
        rows.slice(start, size).write_parquet(path)
        pairs.append((path, f"work/{path.name}"))
    return pairs


def plan(scratch: Path, bucket: Bucket) -> int:
    """Upload the work shards and ``plan.json`` (the mark that planning is done); returns the texts."""
    bucket.create()
    if bucket.files("plan.json"):
        return json.loads(_get(bucket, "plan.json", scratch).read_text())["texts"]
    rev = hub.revision_of(REPO, RELEASE)
    hub.snapshot(REPO, rev, scratch / "release", ["data/*"])
    rows = selection(pl.scan_parquet(scratch / "release" / "data" / "*" / "*" / "*.parquet"))
    texts = rows.collect(engine="streaming")
    bucket.put(work_shards(texts, scratch / "work"))
    summary = {"texts": texts.height, "release": rev}
    bucket.put([(json.dumps(summary).encode(), "plan.json")])
    shutil.rmtree(scratch)
    return texts.height


def _get(bucket: Bucket, path: str, scratch: Path) -> Path:
    local = scratch / path
    bucket.get([(path, local)])
    return local


def pending(bucket: Bucket) -> str:
    """The dispatcher's work items (``g5k-dispatch``): one, the work shards left and the shards done."""
    item = {
        "key": "embed",
        "memory_gb": MEMORY_GB,
        "remaining": len(bucket.files("work/")),
        "fp16_ok": True,  # a GPU without bf16 computes in float32 instead (see ``torch_dtype``)
        "done": len(bucket.files("results/")),
    }
    return json.dumps([item])


def torch_dtype(dtype: str | None) -> str:
    """bf16 as for ``embeddings``; float32 where the dispatcher reports no bf16 (float16 may overflow)."""
    return "float32" if dtype == "float16" else "bfloat16"


def vectors_table(work: pl.DataFrame, vectors: np.ndarray) -> pa.Table:
    """``id``, ``lat``, ``lon`` and ``vector`` (fixed-size list of float16), as in ``embeddings``."""
    flat = pa.array(vectors.astype(np.float16).reshape(-1), type=pa.float16())
    table = work.select("id", "lat", "lon").to_arrow()
    return table.append_column("vector", pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1]))


def embed_shard(encode: Encode, work: pl.DataFrame) -> pa.Table:
    return vectors_table(work, encode(work["text"].to_list()))


def encoder(dtype: str) -> Encode:  # pragma: no cover (GPU job)
    from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]

    model = SentenceTransformer(themes.MODEL, device="cuda", model_kwargs={"dtype": dtype})
    model.max_seq_length = themes.MAX_TOKENS

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, prompt=themes.PROMPT, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True
        )

    return encode


def work(scratch: Path, offset: int, budget_s: float) -> int:  # pragma: no cover (GPU job)
    encode = encoder(torch_dtype(os.environ.get("DTYPE")))
    return workqueue.Worker(Bucket(BUCKET), partial(embed_shard, encode), scratch, offset, budget_s).run()


def result_names(paths: Sequence[str]) -> list[tuple[str, str]]:
    """Each result shard and its name in ``embeddings_en`` (``train-00003-of-00412.parquet``)."""
    return [(path, themes.shard_name(i, len(paths))) for i, path in enumerate(sorted(paths))]


def fetch(out: Path, work_dir: Path, bucket: Bucket) -> None:
    """The release and its card context, then the vectors as ``out/embeddings_en``."""
    if bucket.files("work/"):
        raise RuntimeError("work shards are left: embedding is not finished")
    fuse.fetch_corpus(REPO, hub.revision_of(REPO, RELEASE), out, work_dir)
    hub.snapshot(REPO, hub.revision_of(REPO, RELEASE), out, [f"{f}/*" for f in extras.EXTRAS_FOLDERS])
    extras.restore_context(work_dir)
    pairs = result_names(bucket.files("results/"))
    bucket.get([(path, out / FOLDER / name) for path, name in pairs])


def _checks(out: Path) -> dict[str, pl.LazyFrame]:
    """Each check of ``embeddings_en`` against the release, as a count of the rows that fail it."""
    data = pl.scan_parquet(out / "data" / "*" / "*" / "*.parquet")
    translated = data.filter(pl.col("landuse") & (pl.col("translation") == "nllb"))
    vectors = pl.scan_parquet(out / FOLDER / "*.parquet").select("id", "lat", "lon")
    placed = vectors.join(translated.select("id", "text_en", "lat", "lon"), on="id", suffix="_row")
    moved = (pl.col("lat") != pl.col("lat_row")) | (pl.col("lon") != pl.col("lon_row"))
    missing = translated.select("text_en").join(placed.select("text_en"), on="text_en", how="anti")
    return {
        "vectors sharing an id": vectors.select(pl.len() - pl.col("id").n_unique()),
        "vectors of no relevant translated sentence": vectors.join(translated, on="id", how="anti").select(
            pl.len()
        ),
        "vectors away from their sentence": placed.filter(moved).select(pl.len()),
        "vectors sharing a translation": placed.select(pl.len() - pl.col("text_en").n_unique()),
        "translations without a vector": missing.select(pl.col("text_en").n_unique()),
    }


def check(out: Path) -> list[str]:
    """What is wrong with ``embeddings_en`` against the release in ``out``; empty when it fits."""
    counts = {name: lazy.collect(engine="streaming").item() for name, lazy in _checks(out).items()}
    return [f"{name}: {n:,}" for name, n in counts.items() if n]


def assemble(out: Path, work_dir: Path, bucket: Bucket) -> None:
    """Fetch the release and the vectors, and refuse them unless they fit."""
    fetch(out, work_dir, bucket)
    if problems := check(out):
        raise ValueError("; ".join(problems))


def propose(out: Path) -> str:
    """Open a pull request with ``embeddings_en`` and the new card; the release's own files are left out."""
    for folder in (*RELEASE_FOLDERS, "assets"):
        shutil.rmtree(out / folder, ignore_errors=True)
    return hub.propose(out, REPO, PR_TITLE)


def finalize(number: int, bucket: Bucket) -> None:
    """Merge the reviewed pull request, tag the release, delete the bucket."""
    hub.merge(REPO, number)
    hub.tag(REPO, NEXT_RELEASE)
    bucket.remove()
