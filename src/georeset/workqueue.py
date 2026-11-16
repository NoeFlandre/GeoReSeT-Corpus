"""Work shards in a bucket, shared by GPU jobs on any number of nodes.

A job claims a shard (a ``claims/`` file named with its start time), uploads its result under ``results/``,
then deletes the work shard and its claim, which frees their storage at once. A claim older than
``CLAIM_S`` belongs to a job that died; its shard is taken again.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from georeset.translate.store import Bucket

CLAIM_S = 75 * 60  # longer than a one-hour job


def shard_name(path: str) -> str:
    """``work/fra_Latn/00003.parquet`` -> ``fra_Latn/00003``."""
    return path.removeprefix("work/").removesuffix(".parquet")


def claims(paths: list[str], now: float) -> set[str]:
    """Shards claimed less than ``CLAIM_S`` ago (claim files ``claims/<shard>.<start time>``)."""
    live = set()
    for path in paths:
        shard, _, started = path.removeprefix("claims/").rpartition(".")
        if now - int(started) < CLAIM_S:
            live.add(shard)
    return live


def choose(work: list[str], taken: set[str], offset: int) -> str | None:
    """A shard not ``taken`` (claimed or done), starting ``offset`` into the list so that jobs spread out."""
    free = [shard_name(p) for p in work if shard_name(p) not in taken]
    return free[offset % len(free)] if free else None


def may_start(elapsed: float, longest: float, budget: float) -> bool:
    """Start another shard only if one as long as the longest so far still fits in the budget."""
    return elapsed + longest < budget


@dataclass
class Worker:
    bucket: Bucket
    transform: Callable[[pl.DataFrame], pl.DataFrame | pa.Table]  # a work shard -> its result
    scratch: Path
    offset: int
    budget_s: float
    clock: Callable[[], float] = time.time

    def next_shard(self) -> str | None:
        claimed = claims(self.bucket.files("claims/"), self.clock())
        done = {shard_name(p.replace("results/", "work/", 1)) for p in self.bucket.files("results/")}
        return choose(self.bucket.files("work/"), claimed | done, self.offset)

    def run(self) -> int:
        """Process shards while time allows; returns how many were done."""
        start, longest, done = self.clock(), 0.0, 0
        while may_start(self.clock() - start, longest, self.budget_s):
            shard = self.next_shard()
            if shard is None:
                break
            began = self.clock()
            self.process(shard, int(began))
            longest, done = max(longest, self.clock() - began), done + 1
        return done

    def process(self, shard: str, started: int) -> None:
        claim = f"claims/{shard}.{started}"
        self.bucket.put([(b"", claim)])
        local = self.scratch / "work.parquet"
        self.bucket.get([(f"work/{shard}.parquet", local)])
        result = self.scratch / "result.parquet"
        made = self.transform(pl.read_parquet(local))
        table = made.to_arrow() if isinstance(made, pl.DataFrame) else made
        pq.write_table(table, result, compression="zstd")
        self.bucket.put([(result, f"results/{shard}.parquet")])
        self.bucket.delete([f"work/{shard}.parquet", claim])
        print(f"{shard}: {table.num_rows} texts in {self.clock() - started:.0f} s", flush=True)
