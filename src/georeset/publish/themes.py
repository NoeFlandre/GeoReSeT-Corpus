"""What places talk about: PCA of the mean text embedding per H3 cell, one per corpus.

Three corpora: the relevant sentences (``LANDUSE``), the same in English only (``ENGLISH``, on the same
vectors) and the scene descriptions (``SCENES``). ``write_sentences`` splits a corpus' texts into its
shards; ``scripts/embed.py`` embeds each into the same-named file of the corpus' folder (``id``, ``lat``,
``lon``, ``vector``), row for row. ``compute`` reads the shards twice: the texts' H3 cells, then the cells'
mean vectors, whose PCA it fits. Each text counts once, in its cell of ``geo.PCA_RESOLUTION``. ``render``
turns its outputs into the maps.
"""

import json
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from georeset.publish import card, figures, geo, stats

COMPONENTS = 3
MODEL = "microsoft/harrier-oss-v1-0.6b"  # as in scripts/embed.py
PROMPT = "Instruct: Identify the land use or land cover described\nQuery: "
MAX_TOKENS = 512  # longest input, as in scripts/embed.py
PCS = [f"pc{i}" for i in range(1, COMPONENTS + 1)]
AHEAD = 1  # shards loading in the background while one is processed
CHUNK_ROWS = 65_536  # vectors summed at a time: bounds the memory of a shard
LANDUSE = card.Corpus(
    name="landuse",
    source="data/*/*/*.parquet",
    folder="embeddings",
    what="relevant sentences",
    unit="sentences",
    selection="distinct sentences with `landuse` true (the first `id` keeps a repeated text)",
    shards=128,
    min_texts=20,
    placement="in the H3 cell of its location",
)
SCENES = card.Corpus(
    name="scenes",
    source="scenes/*.parquet",
    folder="scene_embeddings",
    what="scene descriptions",
    unit="descriptions",
    selection="distinct scene descriptions (`text` of `scenes`, `id` its `scene_id`)",
    shards=1,
    min_texts=10,
    placement="once in every H3 cell where its scene has photos",
    photos="scene_photos/*.parquet",
    suffix="_scenes",
)
# The relevant sentences' vectors, English only: what is left of the map without language.
ENGLISH = replace(
    LANDUSE,
    name="english",
    what="relevant sentences in English",
    selection="relevant sentences in English (`language` `eng_Latn`)",
    only_language="eng_Latn",
    suffix="_english",
)
# The relevant sentences in English, original and translated, pooled. Original English vectors are LANDUSE's
# (``embeddings``), translations are ``embeddings_en``.
ENGLISH_POOL = replace(
    LANDUSE,
    name="english_pool",
    what="relevant sentences in English, original and translated",
    selection="relevant sentences in English: those labelled so, and the English translations of the rest",
    only_language="",
    suffix="_english_pool",
)
CORPORA = {corpus.name: corpus for corpus in (LANDUSE, ENGLISH, ENGLISH_POOL, SCENES)}
_SELECT = {
    LANDUSE.name: stats.landuse_sentences,
    ENGLISH.name: stats.landuse_sentences,
    SCENES.name: stats.scene_sentences,
}


def shard_name(i: int, shards: int) -> str:
    return f"train-{i:05d}-of-{shards:05d}.parquet"


def write_sentences(data: pl.LazyFrame, out: Path, corpus: card.Corpus = LANDUSE) -> int:
    """Write a corpus' distinct texts (``id``, ``text``, ``language``, ``lat``, ``lon``) in its shards."""
    shards = corpus.shards
    rows = _SELECT[corpus.name](data).collect(engine="streaming")
    out.mkdir(parents=True, exist_ok=True)
    size = -(-rows.height // shards)
    for i in range(shards):
        rows.slice(i * size, size).write_parquet(out / shard_name(i, shards))
    return rows.height


def read_vectors(source) -> tuple[pl.Series, np.ndarray]:
    """An embeddings shard: its ``id`` and its vectors as a float16 matrix (summed in float64 by chunks)."""
    table = pq.read_table(source, columns=["id", "vector"])
    column = table.column("vector").combine_chunks()
    vectors = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1)
    ids = pl.Series("id", table.column("id").combine_chunks())
    return ids, vectors


def _prefetched(load: Callable, items: Iterable, ahead: int = AHEAD) -> Iterator:
    """``load(item)`` for each item, in order, with up to ``ahead`` more loading in the background."""
    with ThreadPoolExecutor(ahead) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(load, item))
            if len(pending) > ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _rows(text: Path, language: str) -> pl.DataFrame:
    """A sentence shard, only its rows in ``language`` if given."""
    rows = pl.read_parquet(text)
    return rows.filter(language=language) if language else rows


def _pairs(
    sentences: list[Path], embeddings: list, languages: list[str]
) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    """Each sentence shard and the vectors of its rows; only the rows in its ``language`` if given."""

    def load(triple: tuple[Path, object, str]) -> tuple[pl.DataFrame, np.ndarray]:
        text, source, language = triple
        rows = pl.read_parquet(text)
        ids, vectors = read_vectors(source)
        if not rows["id"].equals(ids, check_names=False):
            raise ValueError(f"{source} does not match {text} row for row")
        if language:
            keep = (rows["language"] == language).to_numpy()
            rows, vectors = rows.filter(keep), vectors[keep]
        return rows, vectors

    yield from _prefetched(load, zip(sentences, embeddings, languages, strict=True))


@dataclass(frozen=True)
class _Cells:
    """The H3 cells the texts count in, at ``geo.PCA_RESOLUTION``, and the number of texts."""

    ids: np.ndarray  # sorted cell ids
    placements: list[tuple[np.ndarray, np.ndarray]]  # per shard: row index, then index in ``ids``
    texts: np.ndarray  # texts per cell
    total: int  # texts, each counted once


def _placement(rows: pl.DataFrame, pairs: pl.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Each placement's row and H3 cell: the text's own location, or each cell of its photos."""
    if pairs is None:
        cells = geo.h3_cells(rows["lon"].to_numpy(), rows["lat"].to_numpy(), geo.PCA_RESOLUTION)
        return np.arange(rows.height), cells
    placed = rows.select("id").with_row_index("row").join(pairs, on="id")
    return placed["row"].to_numpy(), placed["cell"].to_numpy()


def _locate(sentences: list[Path], languages: list[str], photos: pl.DataFrame | None) -> _Cells:
    """First pass, without vectors: each text's H3 cell, or each cell where its scene has photos."""
    pairs = None if photos is None else stats.photo_cells(photos)
    index, cells, total = [], [], 0
    for text, language in zip(sentences, languages, strict=True):
        rows = _rows(text, language)
        total += rows.height
        shard_index, shard_cells = _placement(rows, pairs)
        index.append(shard_index)
        cells.append(shard_cells)
    ids = np.unique(np.concatenate(cells))
    positions = tuple(np.searchsorted(ids, c) for c in cells)
    texts = np.bincount(np.concatenate(positions), minlength=len(ids))
    return _Cells(ids, list(zip(index, positions, strict=True)), texts, total)


def _cell_means(pairs, cells: _Cells) -> np.ndarray:
    """Second pass: each cell's mean vector, summed ``CHUNK_ROWS`` rows at a time."""
    sums = None
    for (_, vectors), (index, cell) in zip(pairs, cells.placements, strict=True):
        sums = np.zeros((len(cells.ids), vectors.shape[1])) if sums is None else sums
        for start in range(0, len(index), CHUNK_ROWS):
            rows = slice(start, start + CHUNK_ROWS)
            present, part, _ = stats.group_sums(cell[rows], vectors[index[rows]])
            sums[present] += part
    return np.asarray(sums) / cells.texts[:, None]


def _display(cells: _Cells, values: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells with ``minimum`` texts or more: their ids, ``values`` rows and number of texts."""
    kept = cells.texts >= minimum
    return cells.ids[kept], values[kept], cells.texts[kept].astype(np.int64)


def _with_scores(rows: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    """``rows`` with one column per axis of ``PCS``."""
    return rows.with_columns([pl.Series(name, scores[:, i]) for i, name in enumerate(PCS)])


def compute(
    sentences: list[Path],
    embeddings: list,
    work: Path,
    corpus: card.Corpus = LANDUSE,
    photos: pl.DataFrame | None = None,
    languages: list[str] | None = None,
) -> Path:
    """Each cell's mean vector, then the PCA of the cells with enough texts, in ``work/themes/<corpus>``.

    ``languages``: the language each sentence shard is filtered to (empty: all its rows); by default, the
    corpus' ``only_language`` for every shard.

    ``cells.parquet`` keeps every cell's scores on the axes, for the explorer Space.

    ``photos`` (``scene_id``, ``lat``, ``lon``): each text counts once in every cell of its photos,
    instead of in its own location's.
    """
    languages = languages or [corpus.only_language] * len(sentences)
    cells = _locate(sentences, languages, photos)
    means = _cell_means(_pairs(sentences, embeddings, languages), cells)
    kept, display_means, texts = _display(cells, means, corpus.min_texts)
    fit = stats.pca(display_means, COMPONENTS)
    out = work / "themes" / corpus.name
    out.mkdir(parents=True, exist_ok=True)
    _with_scores(pl.DataFrame({"hexagon": kept, "n": texts}), fit.scores(display_means)).write_parquet(
        out / "hexagons.parquet"
    )
    cell_scores = pl.DataFrame(
        {"cell": cells.ids, "n": cells.texts}, schema={"cell": pl.UInt64, "n": pl.Int64}
    )
    _with_scores(cell_scores, fit.scores(means)).write_parquet(out / "cells.parquet")
    summary = {"sentences": cells.total, "cells": len(cells.ids), "explained": fit.explained.tolist()}
    (out / "summary.json").write_text(json.dumps(summary))
    return out


def _layers(corpus: card.Corpus) -> str:
    return f"Mean embedding of the {corpus.what} per H3 cell (resolution {geo.PCA_RESOLUTION})"


def _grey(minimum: int, corpus: card.Corpus) -> str:
    return f"grey: land with fewer than {minimum} {corpus.unit} per resolution-{geo.PCA_RESOLUTION} cell"


def render(work: Path, assets: Path, corpus: card.Corpus = LANDUSE) -> card.Themes:
    """A corpus: maps of its cells (RGB, then one per axis) in ``assets``; its PCA's facts for the card."""
    out = work / "themes" / corpus.name
    hexagons = pl.read_parquet(out / "hexagons.parquet")
    summary = json.loads((out / "summary.json").read_text())
    scores = hexagons.select(PCS).to_numpy()
    index = hexagons["hexagon"].to_numpy()
    axes = card.Themes(
        corpus=corpus,
        model=MODEL,
        prompt=PROMPT,
        texts=summary["sentences"],
        hexagons=hexagons.height,
        explained=summary["explained"],
    )
    rgb = stats.unit_scale(scores)
    land = geo.land_cells(geo.PCA_RESOLUTION)
    grey = _grey(corpus.min_texts, corpus)
    caption = (
        f"{_layers(corpus)}. Red, green, blue: principal components 1, 2, 3 (2nd to 98th percentile)\n{grey}"
    )
    figures.rgb_map(index, rgb, assets / f"map_pca{corpus.suffix}.png", land, caption)
    for i in range(COMPONENTS):
        label = f"PC{i + 1} ({card.pct(axes.explained[i])} of variance) ({grey})"
        values = scores[:, i]
        path = assets / f"map_pc{i + 1}{corpus.suffix}.png"
        figures.diverging_map(index, values, stats.symmetric_limit(values), label, path, land)
    return axes
