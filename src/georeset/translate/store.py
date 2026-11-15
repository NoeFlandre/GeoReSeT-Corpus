"""The private Hugging Face storage bucket holding the translation's working files.

A bucket, not a repository: files are mutable and deleted without history (so their storage is freed at
once), and writes are not git commits, so many jobs can write without hitting the commit rate limit.

    model/                       NLLB converted for CTranslate2 (deleted at the end)
    plan/<unit>.json             counts of a unit's rows per kind
    routes/<unit>.parquet        website rows whose own language differs from their page's (id, detected)
    work/<unit>/<n>.parquet      texts to translate (key, translation_language, text); deleted once translated
    claims/<unit>/<n>.<time>     a job working on a shard, with its start time (seconds since the epoch)
    results/<unit>/<n>.parquet   translations (key, text_en, translation_flags)
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

BUCKET = "NoeFlandre/georeset-translation"


class Bucket:
    def __init__(self, bucket_id: str = BUCKET, api: HfApi | None = None):
        self.id, self.api = bucket_id, api or HfApi()

    def create(self) -> None:
        self.api.create_bucket(self.id, private=True, exist_ok=True)

    def files(self, prefix: str) -> list[str]:
        """Paths of the files under ``prefix`` (recursive)."""
        entries = self.api.list_bucket_tree(self.id, prefix=prefix, recursive=True)
        return sorted(entry.path for entry in entries if entry.type == "file")

    def put(self, files: Sequence[tuple[Path | bytes, str]]) -> None:
        """Upload ``(local file or bytes, path in the bucket)`` pairs."""
        pairs: list[Any] = list(files)
        self.api.batch_bucket_files(self.id, add=pairs)

    def delete(self, paths: list[str]) -> None:
        if paths:
            self.api.batch_bucket_files(self.id, delete=paths)

    def get(self, files: Sequence[tuple[str, Path]]) -> None:
        """Download ``(path in the bucket, local file)`` pairs."""
        pairs: list[Any] = list(files)
        self.api.download_bucket_files(self.id, pairs, raise_on_missing_files=True)

    def remove(self) -> None:
        """Delete the whole bucket and everything left in it."""
        self.api.delete_bucket(self.id, missing_ok=True)
