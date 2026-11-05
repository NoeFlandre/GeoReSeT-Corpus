# /// script
# requires-python = ">=3.12"
# dependencies = ["sentence-transformers>=5.1", "torch>=2.8", "pyarrow>=21", "huggingface-hub>=1.0"]
# ///
"""Embed the sentence shards of ``georeset.publish.themes`` on a GPU and upload each to the Hub.

    uv run scripts/embed.py SENTENCES_DIR --worker 0 --workers 4

Worker ``k`` starts at the ``k``-th of ``workers`` equal slices of the shards and goes round all of them,
skipping those already in ``--folder`` on the Hub: any number of jobs share the work,
and a crashed job resumes where it stopped. Near the end, give each missing shard its own job (a directory
holding just that shard): workers only see uploaded shards, so they converge on the same unfinished one.
Vectors are L2-normalised and stored as float16.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

REPO = "NoeFlandre/GeoReSeT-Corpus-Fused"
FOLDER = "embeddings"
# Same as georeset.publish.themes; kept here so the script runs without the package.
MODEL = "microsoft/harrier-oss-v1-0.6b"
PROMPT = "Instruct: Identify the land use or land cover described\nQuery: "
MAX_TOKENS = 512


def done(api: HfApi, folder: str) -> set[str]:
    files = api.list_repo_files(REPO, repo_type="dataset")
    return {Path(f).name for f in files if f.startswith(f"{folder}/")}


def embed(model: SentenceTransformer, shard: Path, out: Path, batch: int) -> int:
    rows = pq.read_table(shard, columns=["id", "lat", "lon", "text"])
    vectors = model.encode(
        rows.column("text").to_pylist(),
        prompt=PROMPT,
        batch_size=batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    flat = pa.array(vectors.reshape(-1), type=pa.float16())
    table = rows.drop_columns("text").append_column(
        "vector", pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1])
    )
    pq.write_table(table, out, compression="zstd")
    return table.num_rows


def upload(api: HfApi, path: Path, folder: str, name: str, tries: int = 8) -> None:
    for attempt in range(tries):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{folder}/{name}",
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Embeddings: {folder}/{name}",
            )
            return
        except Exception as error:  # concurrent commits from other workers, transient network errors
            if attempt == tries - 1:
                raise
            print(f"upload {name} failed ({error}), retrying", flush=True)
            time.sleep(30 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--folder", default=FOLDER, help="Folder of the dataset repo to upload to.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Embed this worker's first shard, upload nothing."
    )
    args = parser.parse_args()
    api = HfApi()
    shards = sorted(args.sentences.glob("train-*.parquet"))
    start = len(shards) * args.worker // args.workers
    shards = shards[start:] + shards[:start]
    todo = shards[:1] if args.dry_run else [s for s in shards if s.name not in done(api, args.folder)]
    print(f"{len(todo)} of {len(shards)} shards to embed", flush=True)
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"dtype": "bfloat16"})
    model.max_seq_length = MAX_TOKENS
    with tempfile.TemporaryDirectory() as scratch:
        for shard in todo:
            if not args.dry_run and shard.name in done(api, args.folder):
                continue
            start = time.time()
            out = Path(scratch) / shard.name
            n = embed(model, shard, out, args.batch)
            seconds = time.time() - start
            print(
                f"{shard.name}: {n} sentences in {seconds:.0f} s, {out.stat().st_size / 1e6:.0f} MB",
                flush=True,
            )
            if not args.dry_run:
                upload(api, out, args.folder, shard.name)
            out.unlink()


if __name__ == "__main__":
    main()
