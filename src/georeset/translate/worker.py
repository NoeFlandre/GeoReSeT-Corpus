"""Translate work shards on a GPU until the job's time budget runs out.

Settings are those of the benchmark that chose the model (``llms-xx-to-en-translation``): greedy decoding,
an output budget of 3x the source tokens clamped to [32, 2048] (``xx_to_en.lengths``), float16, and its
failure checks (``xx_to_en.failures``) with GlotLID v4 for the output language.

Jobs share the work shards through ``georeset.workqueue``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl
from xx_to_en.failures import detect
from xx_to_en.lengths import max_new_tokens

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
