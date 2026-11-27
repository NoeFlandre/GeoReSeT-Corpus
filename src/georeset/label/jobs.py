"""Label a sample of the GeoReSeT corpus with the benchmark's model, through a private bucket.

plan    (CPU) sample rows, render one prompt per distinct text and tokenize it; upload both
work    (GPU) label the distinct texts with SGLang and the DSpark draft; upload the generations
report  compare each sampled row's decision with its existing ``landuse`` label
remove  delete the bucket

Nothing is written to the dataset: its existing labels are only read.
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

from georeset.label import config, prompt, tokenizer, worker
from georeset.label.engine import SGLangEngine
from georeset.label.parse import parse_generation
from georeset.label.records import Generation
from georeset.label.text import sha256_text
from georeset.translate.store import Bucket

REPO = "NoeFlandre/GeoReSeT-Corpus"
SAMPLE = "work/sample.parquet"  # id, text_sha256, landuse of every sampled row
TEXTS = "work/texts.parquet"  # text_sha256, text, input_ids of every distinct sampled text
RESULTS = "results/texts.parquet"  # one generation per distinct text, with its decision

Encoder = Callable[[list[str]], list[list[int]]]  # rendered prompts -> token ids


def plan(rows: pl.DataFrame, bucket: Bucket, scratch: Path, encode: Encoder) -> None:
    """Upload the sampled rows and one tokenized prompt per distinct text."""
    template = prompt.text()
    texts = rows["text"].unique(maintain_order=True).to_list()
    sample = pl.DataFrame(
        {
            "id": rows["id"],
            "text_sha256": [sha256_text(text) for text in rows["text"]],
            "landuse": rows["landuse"],
        }
    )
    distinct = pl.DataFrame(
        {
            "text_sha256": [sha256_text(text) for text in texts],
            "text": texts,
            "input_ids": encode([prompt.render(template, text) for text in texts]),
        }
    )
    _put(bucket, scratch, sample, SAMPLE)
    _put(bucket, scratch, distinct, TEXTS)


def work(bucket: Bucket, scratch: Path, engine: worker.AsyncEngine) -> None:
    """Generate every distinct text of the plan with ``engine`` and upload the generations."""
    shard = _get(bucket, scratch, TEXTS)
    items = [
        worker.Item(row["text_sha256"], row["text"], row["input_ids"]) for row in shard.iter_rows(named=True)
    ]
    generations = asyncio.run(worker.label(engine, items))
    _put(bucket, scratch, _results(generations), RESULTS)


def report(bucket: Bucket, scratch: Path) -> dict[str, Any]:
    """Agreement of each sampled row's decision with its existing ``landuse`` label."""
    sample = _get(bucket, scratch, SAMPLE)
    results = _get(bucket, scratch, RESULTS).select("text_sha256", "decision")
    joined = sample.join(results, on="text_sha256", how="left")
    expected = pl.when(pl.col("landuse")).then(pl.lit("yes")).otherwise(pl.lit("no"))
    agrees = (pl.col("decision") == expected).fill_null(False)
    return {
        "rows": joined.height,
        "matches": joined.filter(agrees).height,
        "failed": joined.filter(pl.col("decision") == "failed").height,
        "mismatches": joined.filter(~agrees).select("id", "decision", "landuse").to_dicts(),
    }


def remove(bucket: Bucket) -> None:
    """Delete the bucket and everything left in it."""
    bucket.remove()


def plan_corpus(bucket: Bucket, scratch: Path, sample: int, seed: int) -> None:  # pragma: no cover
    """``plan`` on a sample of the corpus, tokenized with the benchmark's chat template."""
    bucket.create()
    encode = tokenizer.chat_encoder(config.MODEL_ID, config.MODEL_REVISION)
    plan(_corpus_sample(sample, seed), bucket, scratch, encode)


def work_on_gpu(bucket: Bucket, scratch: Path) -> None:  # pragma: no cover
    """``work`` with the benchmark's SGLang engine; GPU memory is released at the end."""
    engine = SGLangEngine(config.engine_kwargs(), config.SAMPLING)
    try:
        work(bucket, scratch, engine)
    finally:
        engine.shutdown()


def _corpus_sample(n: int, seed: int) -> pl.DataFrame:  # pragma: no cover
    """``n`` corpus rows, half labelled relevant and half not, from four data files spread over the repo."""
    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    picked = files[:: max(1, len(files) // 4)][:4]
    frames = [
        pl.read_parquet(hf_hub_download(REPO, name, repo_type="dataset"), columns=["id", "text", "landuse"])
        for name in picked
    ]
    frame = pl.concat(frames)
    half = n // 2
    relevant = frame.filter(pl.col("landuse")).sample(half, seed=seed)
    other = frame.filter(~pl.col("landuse")).sample(n - half, seed=seed)
    return pl.concat([relevant, other])


def _results(generations: list[Generation]) -> pl.DataFrame:
    rows = []
    for generation in generations:
        verdict = parse_generation(generation.raw_output, truncated=generation.truncated)
        rows.append(
            {
                **asdict(generation),
                "decision": str(verdict.decision),
                "failure": None if verdict.failure is None else str(verdict.failure),
            }
        )
    return pl.from_dicts(
        rows,
        schema_overrides={
            "verify_steps": pl.Int64,
            "accepted_drafts": pl.Int64,
            "proposed_drafts": pl.Int64,
            "latency_s": pl.Float64,
            "failure": pl.String,
        },
    )


def _put(bucket: Bucket, scratch: Path, frame: pl.DataFrame, path: str) -> None:
    local = scratch / path
    local.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(local)
    bucket.put([(local, path)])


def _get(bucket: Bucket, scratch: Path, path: str) -> pl.DataFrame:
    local = scratch / "download" / path
    local.parent.mkdir(parents=True, exist_ok=True)
    bucket.get([(path, local)])
    return pl.read_parquet(local)
