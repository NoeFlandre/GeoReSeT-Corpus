"""Round 1: the corpus's existing labels, with the raw output the labelling recorded for each text.

The decision is the existing ``landuse`` label of each row. The raw output comes from the generations of
the label repository that labelled the row (its group), joined on the text's sha256. Raw outputs are stored
once per group and text, not once per row, since many rows share a text. Each raw output is parsed again,
so the agreement between the parser and the label is reported rather than assumed.
"""

import polars as pl

from georeset.label.parse import parse_generation

MISSING = "missing_generation"
KEYS = ["group", "text_sha256"]


def _judged(corpus: pl.DataFrame, generations: pl.DataFrame) -> pl.DataFrame:
    """One row per corpus row, with its label, its generation and the parser's verdict on it."""
    joined = corpus.select("id", *KEYS, "landuse").join(
        generations.select(*KEYS, "raw_output", "truncated"), on=KEYS, how="left"
    )
    model, failure = [], []
    for raw, truncated in zip(joined["raw_output"].to_list(), joined["truncated"].to_list(), strict=True):
        if raw is None:
            model.append(None)
            failure.append(MISSING)
            continue
        verdict = parse_generation(raw, truncated=bool(truncated))
        model.append(str(verdict.decision))
        failure.append(None if verdict.failure is None else str(verdict.failure))
    return joined.with_columns(
        decision=pl.when(pl.col("landuse")).then(pl.lit("yes")).otherwise(pl.lit("no")),
        model_decision=pl.Series(model, dtype=pl.String),
        failure=pl.Series(failure, dtype=pl.String),
    )


def round_rows(corpus: pl.DataFrame, generations: pl.DataFrame) -> pl.DataFrame:
    """The round file: each corpus row's id and text sha256, its decision and any failure."""
    return _judged(corpus, generations).select("id", "text_sha256", "decision", "failure")


def raw_rows(corpus: pl.DataFrame, generations: pl.DataFrame) -> pl.DataFrame:
    """The round's generations: each (group, text) the corpus uses, once, with its raw output."""
    used = corpus.select(*KEYS).unique()
    return used.join(
        generations.select(*KEYS, "raw_output", "finish_reason", "truncated"),
        on=KEYS,
        how="inner",
    )


def agreement(corpus: pl.DataFrame, generations: pl.DataFrame) -> dict[str, int]:
    """How many raw outputs the parser turns back into the existing label, and the rest."""
    judged = _judged(corpus, generations)
    parsed = judged.filter(pl.col("raw_output").is_not_null())
    agree = parsed.filter(pl.col("model_decision") == pl.col("decision")).height
    return {
        "rows": judged.height,
        "parsed": parsed.height,
        "agree": agree,
        "disagree": parsed.height - agree,
        "missing": judged.height - parsed.height,
    }
