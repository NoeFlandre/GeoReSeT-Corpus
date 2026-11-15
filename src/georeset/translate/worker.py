"""Translate work shards on a GPU until the job's time budget runs out.

Settings are those of the benchmark that chose the model (``llms-xx-to-en-translation``): greedy decoding,
an output budget of 3x the source tokens clamped to [32, 2048] (``xx_to_en.lengths``), float16, and its
failure checks (``xx_to_en.failures``) with GlotLID v4 for the output language.

Jobs coordinate through the bucket only: a job claims a shard (a ``claims/`` file named with its start
time), uploads the result, then deletes the work shard and its claim, which frees their storage at once.
A claim older than ``CLAIM_S`` belongs to a job that died; its shard is taken again.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl
from xx_to_en.failures import detect
from xx_to_en.lengths import max_new_tokens

from georeset.translate.store import Bucket

CLAIM_S = 75 * 60  # longer than a one-hour job
CHUNK = 1024  # sentences per call to the model, so that each call's budget fits its sentences
MAX_INPUT_TOKENS = 1024  # NLLB's positions; longer inputs are cut and flagged
# CTranslate2 has no decoder position past 1023 (transformers extends its sinusoids): the benchmark's
# budget (up to 2,048) is capped here; an output that reaches the cap is flagged ``truncated``.
MAX_OUTPUT_TOKENS = 1020


@dataclass(frozen=True)
class Output:
    text: str
    tokens: int  # generated tokens, the target language token included


class Engine(Protocol):
    def count(self, texts: Sequence[str], language: str) -> list[int]:
        """Source tokens of each text, without the special tokens."""
        ...

    def translate(self, texts: Sequence[str], language: str, budgets: Sequence[int]) -> list[Output]:
        """Greedy translations into English, each cut to its own budget of generated tokens."""
        ...


Identify = Callable[[list[str]], list[str]]


def flags(source: str, output: Output, budget: int, source_tokens: int, identify_one: str) -> list[str]:
    """Names of the failed checks: the benchmark's, ``truncated`` (budget reached) and ``input_truncated``."""
    found = detect(source, output.text, identify=lambda _: identify_one)
    names = [name for name in _CHECKS if getattr(found, name)]
    names += ["truncated"] if output.tokens >= budget else []
    return names + (["input_truncated"] if source_tokens > MAX_INPUT_TOKENS - 2 else [])


_CHECKS = ("empty", "repetition_loop", "length_ratio_outlier", "untranslated_copy", "wrong_language")


def translate_group(
    engine: Engine, texts: list[str], language: str
) -> tuple[list[Output], list[int], list[int]]:
    """Outputs, budgets and source token counts for texts of one language, chunk by chunk."""
    counts = engine.count(texts, language)
    budgets = [min(max_new_tokens(n), MAX_OUTPUT_TOKENS) for n in counts]
    outputs: list[Output] = []
    for start in range(0, len(texts), CHUNK):
        chunk = slice(start, start + CHUNK)
        outputs += engine.translate(texts[chunk], language, budgets[chunk])
    return outputs, budgets, counts


def translate_shard(engine: Engine, identify: Identify, work: pl.DataFrame) -> pl.DataFrame:
    """``(key, text_en, translation_flags)`` for a work shard."""
    parts = []
    for (language,), group in work.group_by("translation_language", maintain_order=True):
        texts = group["text"].to_list()
        outputs, budgets, counts = translate_group(engine, texts, str(language))
        detected = identify([o.text for o in outputs])
        names = [flags(*row) for row in zip(texts, outputs, budgets, counts, detected, strict=True)]
        parts.append(
            pl.DataFrame(
                {"key": group["key"], "text_en": [o.text for o in outputs], "translation_flags": names},
                schema={"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)},
            )
        )
    return pl.concat(parts)


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
    engine: Engine
    identify: Identify
    scratch: Path
    offset: int
    budget_s: float
    clock: Callable[[], float] = time.time

    def next_shard(self) -> str | None:
        claimed = claims(self.bucket.files("claims/"), self.clock())
        done = {shard_name(p.replace("results/", "work/", 1)) for p in self.bucket.files("results/")}
        return choose(self.bucket.files("work/"), claimed | done, self.offset)

    def run(self) -> int:
        """Translate shards while time allows; returns how many were done."""
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
        translated = translate_shard(self.engine, self.identify, pl.read_parquet(local))
        translated.write_parquet(result, compression="zstd")  # pragma: no mutate (zstd is the default)
        self.bucket.put([(result, f"results/{shard}.parquet")])
        self.bucket.delete([f"work/{shard}.parquet", claim])
        print(f"{shard}: {translated.height} texts in {self.clock() - started:.0f} s", flush=True)
