"""Add the translation columns to a copy of the published shards, leaving every existing column as it is.

Each shard is read with pyarrow, routed as the plan routed it (same code, same GlotLID overrides), joined
to the translations by key, and written again under its own name with the four new columns appended, its
row groups and schema metadata kept. The new file is read back and its original columns compared to the
old ones before it replaces it.
"""

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from georeset.build.write import COMPRESSION
from georeset.translate import route

NEW_COLUMNS = ("text_en", "translation", "translation_language", "translation_flags")
TRANSLATIONS_SCHEMA = {"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)}


class MissingTranslation(RuntimeError):
    """A row routed to NLLB has no translation: the plan and the results disagree."""


def translation_columns(
    frame: pl.DataFrame, overrides: pl.DataFrame, translations: pl.DataFrame
) -> pl.DataFrame:
    """The four new columns for ``frame``'s rows, in order."""
    if "status" not in frame.columns:
        frame = frame.with_columns(status=pl.lit(None, dtype=pl.String))
    routed = route.routed(frame.select(route.COLUMNS).lazy(), overrides.lazy())
    joined = routed.join(translations.lazy(), on="key", how="left", maintain_order="left").collect()
    is_nllb = pl.col("translation") == route.NLLB
    missing = joined.filter(is_nllb & pl.col("text_en").is_null()).height
    if missing:
        raise MissingTranslation(f"{missing} rows")
    return joined.select(
        text_en=pl.when(pl.col("translation") == route.ORIGINAL).then("text").otherwise("text_en"),
        translation="translation",
        translation_language="translation_language",
        translation_flags=pl.when(is_nllb)
        .then("translation_flags")
        .otherwise(pl.lit([], dtype=pl.List(pl.String))),
    )


def with_translations(table: pa.Table, columns: pl.DataFrame) -> pa.Table:
    """``table`` with ``columns`` appended; its own columns and schema metadata untouched."""
    out = table
    for name, column in zip(columns.columns, columns.to_arrow().columns, strict=True):
        out = out.append_column(name, column)
    return out.replace_schema_metadata(table.schema.metadata)


def same_originals(old: pa.Table, new: pa.Table) -> bool:
    """True when ``new`` starts with exactly ``old``'s columns and values, then the new columns."""
    head = new.select(old.column_names)
    return (
        new.column_names == [*old.column_names, *NEW_COLUMNS]
        and head.equals(old)
        and new.num_rows == old.num_rows
    )


def rewrite(path: Path, overrides: pl.DataFrame, translations: pl.DataFrame) -> None:
    """Replace the shard at ``path`` with its translated version, after checking it."""
    old = pq.read_table(path)
    row_group = pq.ParquetFile(path).metadata.row_group(0).num_rows if old.num_rows else 1
    present = [name for name in route.COLUMNS if name in old.column_names]
    columns = translation_columns(pl.DataFrame(old.select(present)), overrides, translations)
    tmp = path.with_name(path.name + ".tmp")  # pragma: no mutate (private scratch name)
    pq.write_table(with_translations(old, columns), tmp, row_group_size=row_group, compression=COMPRESSION)
    if not same_originals(old, pq.read_table(tmp)):
        tmp.unlink()
        raise RuntimeError(f"{path}: original columns changed")
    tmp.replace(path)


def load_translations(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(schema=TRANSLATIONS_SCHEMA)
    return pl.concat([pl.read_parquet(p) for p in paths]).unique("key", maintain_order=True)
