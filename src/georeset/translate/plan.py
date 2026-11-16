"""Plan one unit (a language folder of the release, or ``unlabelled``): what to translate, deduplicated.

For each published shard of the unit: detect the language of its website sentences, route every row,
keep the distinct texts NLLB must translate. The unit's texts are sorted by language then length (so that
batches pad little) and cut into work shards of about ``SHARD_CHARS`` characters.
"""

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl

from georeset.translate import route

UNLABELLED = "unlabelled"
SHARD_CHARS = 1_500_000
WORK_COLUMNS = ("key", "translation_language", "text")
Identify = Callable[[list[str]], tuple[list[str], list[float]]]


def units(paths: Iterable[str]) -> list[str]:
    """Language folders of ``data/<source>/<language>/`` (each once), then ``unlabelled``."""
    languages = {Path(p).parts[2] for p in paths if p.startswith("data/") and len(Path(p).parts) == 4}
    return [*sorted(languages), UNLABELLED]


def pattern(unit: str) -> str:
    return f"{UNLABELLED}/*.parquet" if unit == UNLABELLED else f"data/*/{unit}/*.parquet"


def rows(path: Path) -> pl.DataFrame:
    """A published shard's ``route.COLUMNS``; labelled shards have no ``status``."""
    frame = pl.read_parquet(path)
    if "status" not in frame.columns:
        frame = frame.with_columns(status=pl.lit(None, dtype=pl.String))
    return frame.select(route.COLUMNS)


def detect(frame: pl.DataFrame, identify: Identify) -> pl.DataFrame:
    """Overrides for the website rows of ``frame``."""
    website = frame.filter(pl.col("source") == "website")
    labels, probabilities = identify(website["text"].to_list())
    return route.overrides(website, labels, probabilities)


def to_translate(routed: pl.DataFrame) -> pl.DataFrame:
    """The distinct texts NLLB must translate."""
    keep = routed.filter(pl.col("translation") == route.NLLB).unique("key", maintain_order=True)
    return keep.select("key", "translation_language", text="model_text")


def ordered(work: pl.DataFrame) -> pl.DataFrame:
    return work.sort(pl.col("translation_language"), pl.col("text").str.len_chars(), pl.col("key"))


def shards(work: pl.DataFrame, chars: int = SHARD_CHARS) -> list[pl.DataFrame]:
    """Consecutive slices of ``work`` holding about ``chars`` characters each."""
    size = work["text"].str.len_chars().cum_sum() - 1
    # pragma: no mutate start (size is unsigned: / then a cast truncates like //)
    labelled = work.with_columns(shard=(size // chars).cast(pl.Int64))
    # pragma: no mutate end
    # pragma: no mutate start (order kept by default)
    return [part.drop("shard") for part in labelled.partition_by("shard", maintain_order=True)]
    # pragma: no mutate end


class Plan:
    """What one unit's shards hold, gathered shard by shard."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.work: list[pl.DataFrame] = []
        self.overrides: list[pl.DataFrame] = []

    def add(self, frame: pl.DataFrame, identify: Identify) -> None:
        detected = detect(frame, identify)
        routed = route.routed(frame.lazy(), detected.lazy()).collect()
        self.counts.update(dict(routed["translation"].value_counts().iter_rows()))
        self.work.append(to_translate(routed))
        self.overrides.append(detected)

    def texts(self) -> pl.DataFrame:
        work = pl.concat(self.work) if self.work else pl.DataFrame(schema=_WORK_SCHEMA)
        # pragma: no mutate start (a key fixes its row, which ordered() sorts)
        return ordered(work.unique("key", maintain_order=True))
        # pragma: no mutate end

    def summary(self, texts: pl.DataFrame) -> dict[str, int]:
        return {**dict(sorted(self.counts.items())), "distinct": texts.height, "characters": _chars(texts)}


_WORK_SCHEMA = {"key": pl.Binary, "translation_language": pl.String, "text": pl.String}


def _chars(texts: pl.DataFrame) -> int:
    return int(texts["text"].str.len_chars().sum())


def write(unit: str, plan: Plan, scratch: Path) -> list[tuple[Path, str]]:
    """Write the unit's files under ``scratch``; returns ``(local, bucket path)`` pairs, summary last."""
    scratch.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    texts = plan.texts()
    files = []
    for i, part in enumerate(shards(texts)):
        files.append(_parquet(part, scratch / f"work-{i:05d}.parquet", f"work/{unit}/{i:05d}.parquet"))
    overrides = pl.concat(plan.overrides) if plan.overrides else pl.DataFrame(schema=_ROUTE_SCHEMA)
    files.append(_parquet(overrides, scratch / "routes.parquet", f"routes/{unit}.parquet"))
    summary = scratch / "plan.json"
    summary.write_text(json.dumps(plan.summary(texts), indent=2))
    return [*files, (summary, f"plan/{unit}.json")]


_ROUTE_SCHEMA = {"id": pl.String, "detected": pl.String}


def _parquet(frame: pl.DataFrame, local: Path, remote: str) -> tuple[Path, str]:
    # pragma: no mutate start (zstd is the default)
    frame.write_parquet(local, compression="zstd")
    # pragma: no mutate end
    return local, remote
