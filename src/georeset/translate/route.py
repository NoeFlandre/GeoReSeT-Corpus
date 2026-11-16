"""How each row gets its English text: kept (already English), translated by NLLB, or left out.

A website row carries its page's language; when GlotLID is confident a sentence is in another language
(an English menu on a Hindi page), that language is used instead. The plan records those overrides, so the
final assembly routes every row exactly as the plan did without running GlotLID again.
"""

import hashlib

import polars as pl

from georeset.translate import languages

THRESHOLD = 0.9  # GlotLID probability above which a website sentence's own language replaces its page's
ORIGINAL, NLLB, UNSUPPORTED, UNSPLIT = "original", "nllb", "unsupported", "unsplit"
COLUMNS = ("id", "text", "language", "source", "status")


def overrides(rows: pl.DataFrame, labels: list[str], probabilities: list[float]) -> pl.DataFrame:
    """``(id, detected)`` for website ``rows`` whose confident GlotLID label differs from their page's."""
    detected = rows.select("id", "language").with_columns(
        detected=pl.Series(labels, dtype=pl.String), probability=pl.Series(probabilities, dtype=pl.Float64)
    )
    confident = (pl.col("probability") >= THRESHOLD) & (pl.col("detected") != pl.col("language"))
    return detected.filter(confident).select("id", "detected")


def key(code: str, text: str) -> bytes:
    """Stable key of a text to translate from ``code``: the first 16 bytes of its SHA-256."""
    return hashlib.sha256(f"{code}\n{text}".encode()).digest()[:16]


def _keys(pairs: pl.Series) -> pl.Series:
    # pragma: no mutate start (bytes give Binary)
    return pl.Series([key(p["code"], p["text"]) for p in pairs], dtype=pl.Binary)
    # pragma: no mutate end


def routed(rows: pl.LazyFrame, detected: pl.LazyFrame) -> pl.LazyFrame:
    """``rows`` (``COLUMNS``) with ``translation``, ``translation_language``, ``model_text`` and ``key``.

    ``key`` is null unless the row goes to NLLB.
    """
    language = pl.coalesce("detected", "language")
    code = languages.nllb(language, pl.col("text"))
    kind = (
        pl.when(language == languages.ENGLISH)
        .then(pl.lit(ORIGINAL))
        .when(pl.col("status") == UNSPLIT)
        .then(pl.lit(UNSPLIT))
        .when(code.is_null())
        .then(pl.lit(UNSUPPORTED))
        .otherwise(pl.lit(NLLB))
    )
    with_kind = (
        rows.join(detected, on="id", how="left", maintain_order="left")
        .with_columns(translation=kind, code=code, model_text=languages.model_text(language, pl.col("text")))
        .with_columns(translation_language=pl.when(pl.col("translation") == NLLB).then("code"))
        .drop("detected", "code")
    )
    pairs = pl.struct(code="translation_language", text="model_text")
    # pragma: no mutate start (_keys gives Binary, with or without return_dtype)
    return with_kind.with_columns(
        key=pl.when(pl.col("translation") == NLLB).then(pairs.map_batches(_keys, return_dtype=pl.Binary))
    )
    # pragma: no mutate end
