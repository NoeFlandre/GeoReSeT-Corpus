"""What places talk about: PCA of the mean text embedding per H3 cell, one per corpus.

Three corpora: the relevant sentences (``LANDUSE``), the same in English only (``ENGLISH``, on the same
vectors) and the scene descriptions (``SCENES``). ``write_sentences`` splits a corpus' texts into its
shards; ``scripts/embed.py`` embeds each into the same-named file of the corpus' folder (``id``, ``lat``,
``lon``, ``vector``), row for row. ``compute`` reads the shards three times: the texts' H3 cells, the
cells' mean vectors (then the display cells' means and their PCA), and each text's scores (axis ends).
``render`` turns its outputs into the maps and the corpus' axes report.
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

from georeset.normalize.languages import display_name
from georeset.publish import card, figures, geo, stats

COMPONENTS = 3
EXTREMES = 10  # texts listed per end of an axis
MIX = 1_000  # texts kept per end, whose languages are counted
MIX_TOP = 5
LANGUAGE_WARNING = 0.5
MODEL = "microsoft/harrier-oss-v1-0.6b"  # as in scripts/embed.py
PROMPT = "Instruct: Identify the land use or land cover described\nQuery: "
PCS = [f"pc{i}" for i in range(1, COMPONENTS + 1)]
AHEAD = 4  # shards loading in the background while one is processed
LANDUSE = card.Corpus(
    name="landuse",
    source="data/*/*/*.parquet",
    folder="embeddings",
    what="relevant sentences",
    unit="sentences",
    selection="distinct sentences with `landuse` true (the first `id` keeps a repeated text)",
    shards=128,
    min_texts=20,
    axes={
        1: ("Cebuano place stubs", "Russian village histories"),
        2: ("airports and stations", "lakes, wetlands and islands"),
        3: ("Cebuano airport stubs", "Iberian church architecture"),
    },
    mixed_languages=True,
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
    axes={
        1: ("paved urban streets", "arid rugged land"),
        2: ("dry open land", "lakes and forests"),
        3: ("lush tropical vegetation", "polar ice and cold seas"),
    },
    mixed_languages=False,
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
    axes={
        1: ("historic city buildings", "lakes and outdoor recreation"),
        2: ("residential amenities", "islands and geology"),
        3: ("airports and cargo hubs", "rural churches and woodland"),
    },
    mixed_languages=False,
    only_language="eng_Latn",
    suffix="_english",
)
CORPORA = {corpus.name: corpus for corpus in (LANDUSE, ENGLISH, SCENES)}
_SELECT = {
    LANDUSE.name: stats.landuse_sentences,
    ENGLISH.name: stats.landuse_sentences,
    SCENES.name: stats.scene_sentences,
}


def shard_name(i: int, shards: int) -> str:
    return f"train-{i:05d}-of-{shards:05d}.parquet"


def write_sentences(
    data: pl.LazyFrame, out: Path, corpus: card.Corpus = LANDUSE, shards: int | None = None
) -> int:
    """Write a corpus' distinct texts (``id``, ``text``, ``language``, ``lat``, ``lon``) in its shards."""
    shards = shards or corpus.shards
    rows = _SELECT[corpus.name](data).collect(engine="streaming")
    out.mkdir(parents=True, exist_ok=True)
    size = -(-rows.height // shards)
    for i in range(shards):
        rows.slice(i * size, size).write_parquet(out / shard_name(i, shards))
    return rows.height


def read_vectors(source, filesystem=None) -> tuple[pl.Series, np.ndarray]:
    """An embeddings shard: its ``id`` and its vectors as a float32 matrix."""
    table = pq.read_table(source, columns=["id", "vector"], filesystem=filesystem)
    column = table.column("vector").combine_chunks()
    vectors = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1)
    ids = pl.Series("id", table.column("id").combine_chunks())
    return ids, vectors.astype(np.float32)


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
    sentences: list[Path], embeddings: list, filesystem, language: str = ""
) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    """Each sentence shard and the vectors of its rows; only the rows in ``language`` if given."""

    def load(pair: tuple[Path, object]) -> tuple[pl.DataFrame, np.ndarray]:
        text, source = pair
        rows = pl.read_parquet(text)
        ids, vectors = read_vectors(source, filesystem)
        if not rows["id"].equals(ids, check_names=False):
            raise ValueError(f"{source} does not match {text} row for row")
        if language:
            keep = (rows["language"] == language).to_numpy()
            rows, vectors = rows.filter(keep), vectors[keep]
        return rows, vectors

    yield from _prefetched(load, zip(sentences, embeddings, strict=True))


@dataclass(frozen=True)
class _Cells:
    """The maps' first layer: the H3 cells the texts count in."""

    ids: np.ndarray  # sorted cell ids
    placements: list[tuple[np.ndarray, np.ndarray]]  # per shard: row index, then index in ``ids``
    texts: np.ndarray  # texts per cell
    languages: pl.DataFrame  # (``cell`` index, ``language``, ``n`` texts)


def _placement(rows: pl.DataFrame, pairs: pl.DataFrame | None) -> tuple[np.ndarray, np.ndarray, pl.Series]:
    """Each placement's row, H3 cell and language: the text's own location, or each cell of its photos."""
    if pairs is None:
        index, cells = np.arange(rows.height), geo.h3_cells(rows["lon"].to_numpy(), rows["lat"].to_numpy())
    else:
        placed = rows.select("id").with_row_index("row").join(pairs, on="id")
        index, cells = placed["row"].to_numpy(), placed["cell"].to_numpy()
    return index, cells, rows["language"].gather(index)


def _languages(cells: tuple[np.ndarray, ...], languages: tuple[pl.Series, ...]) -> pl.DataFrame:
    """Texts per (``cell``, ``language``), from each shard's placements."""
    shards = (pl.DataFrame({"cell": c, "language": lang}) for c, lang in zip(cells, languages, strict=True))
    return pl.concat(shards).group_by("cell", "language").agg(n=pl.len())


def _locate(sentences: list[Path], language: str, photos: pl.DataFrame | None) -> _Cells:
    """First pass, without vectors: each text's H3 cell, or each cell where its scene has photos."""
    pairs = None if photos is None else stats.photo_cells(photos)
    index, cells, languages = zip(
        *(_placement(_rows(text, language), pairs) for text in sentences), strict=True
    )
    ids = np.unique(np.concatenate(cells))
    positions = tuple(np.searchsorted(ids, c) for c in cells)
    texts = np.bincount(np.concatenate(positions), minlength=len(ids))
    return _Cells(ids, list(zip(index, positions, strict=True)), texts, _languages(positions, languages))


def _cell_means(pairs, cells: _Cells) -> np.ndarray:
    """Second pass: each cell's mean vector."""
    sums = None
    for (_, vectors), (index, cell) in zip(pairs, cells.placements, strict=True):
        sums = np.zeros((len(cells.ids), vectors.shape[1])) if sums is None else sums
        present, part, _ = stats.group_sums(cell, vectors[index])
        sums[present] += part
    return np.asarray(sums) / cells.texts[:, None]


def _display(cells: _Cells, values: np.ndarray, minimum: int) -> tuple[np.ndarray, ...]:
    """The display layer: the parent cells with ``minimum`` texts or more.

    Returns those cells, the mean of their child cells' ``values`` (each child counts once), their number
    of texts, and every child's parent.
    """
    parents, sums, size, inverse = stats.parent_sums(cells.ids, values)
    _, texts, _, _ = stats.parent_sums(cells.ids, cells.texts[:, None])
    kept = texts[:, 0] >= minimum
    return parents[kept], sums[kept] / size[kept, None], texts[kept, 0].astype(np.int64), parents[inverse]


def ends(scored: pl.DataFrame, n: int) -> pl.DataFrame:
    """The ``n`` highest and lowest rows on each component: ``pc``, ``end``, ``rank``, row, ``score``."""
    parts = []
    for i, pc in enumerate(PCS, 1):
        for end, rows in zip(
            ("high", "low"), stats.extremes(scored.rename({pc: "score"}), "score", n), strict=True
        ):
            parts.append(
                rows.select("id", "text", "language", "score")
                .with_row_index("rank", 1)
                .with_columns(pc=pl.lit(i), end=pl.lit(end))
            )
    return pl.concat(parts).select("pc", "end", "rank", "id", "text", "language", "score")


def _with_scores(rows: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return rows.with_columns([pl.Series(pc, scores[:, i]) for i, pc in enumerate(PCS)])


def _ends(pairs, fit: stats.Pca) -> tuple[int, pl.DataFrame]:
    """Third pass: the number of texts and each axis' most extreme texts."""
    texts, best = 0, []
    for rows, vectors in pairs:
        texts += rows.height
        best.append(ends(_with_scores(rows, fit.scores(vectors)), MIX))
    return texts, _final_ends(pl.concat(best))


def _final_ends(candidates: pl.DataFrame) -> pl.DataFrame:
    """The ``MIX`` most extreme rows per component and end, ranked, from every shard's own."""
    return pl.concat(
        [
            candidates.filter(pc=i, end=end)
            .sort("score", "id", descending=[end == "high", False])
            .head(MIX)
            .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32))
            for i in range(1, COMPONENTS + 1)
            for end in ("high", "low")
        ]
    )


def compute(
    sentences: list[Path],
    embeddings: list,
    work: Path,
    corpus: card.Corpus = LANDUSE,
    photos: pl.DataFrame | None = None,
    filesystem=None,
) -> Path:
    """Two-layer means (H3 cells, then their parents), their PCA and axis ends, in ``work/themes/<corpus>``.

    ``photos`` (``scene_id``, ``lat``, ``lon``): each text counts once in every cell of its photos,
    instead of in its own location's.
    """
    language = corpus.only_language
    cells = _locate(sentences, language, photos)
    means = _cell_means(_pairs(sentences, embeddings, filesystem, language), cells)
    kept, display_means, texts, parents = _display(cells, means, corpus.min_texts)
    fit = stats.pca(display_means, COMPONENTS)
    hexagons = _with_scores(pl.DataFrame({"hexagon": kept, "n": texts}), fit.scores(display_means))
    mix = cells.languages.with_columns(hexagon=pl.Series(parents).gather(cells.languages["cell"]))
    total, best = _ends(_pairs(sentences, embeddings, filesystem, language), fit)
    out = work / "themes" / corpus.name
    out.mkdir(parents=True, exist_ok=True)
    languages = stats.dominant_language(mix.group_by("hexagon", "language").agg(pl.col("n").sum()))
    hexagons.join(languages, on="hexagon").write_parquet(out / "hexagons.parquet")
    best.write_parquet(out / "ends.parquet")
    summary = {"sentences": total, "cells": len(cells.ids), "explained": fit.explained.tolist()}
    (out / "summary.json").write_text(json.dumps(summary))
    return out


def _share(x: float) -> str:
    return f"{x:.1%}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _mix(rows: pl.DataFrame) -> str:
    mix = stats.language_mix(rows).head(MIX_TOP)
    return ", ".join(f"{display_name(lang)} {_share(share)}" for lang, share in mix.iter_rows())


def _end(rows: pl.DataFrame, title: str, languages: bool) -> list[str]:
    """One end of an axis: its first ``EXTREMES`` texts, with their language mix if languages vary."""
    intro = (
        [f"Languages of the {rows.height:,} sentences at this end: {_mix(rows)}.", ""] if languages else []
    )
    header = (
        "| # | Score | Language | Sentence |\n|---:|---:|---|---|"
        if languages
        else "| # | Score | Text |\n|---:|---:|---|"
    )
    lines = [f"### {title}", "", *intro, header]
    for r in rows.filter(pl.col("rank") <= EXTREMES).iter_rows(named=True):
        language = f" {r['language']} |" if languages else ""
        lines.append(f"| {r['rank']} | {r['score']:.3f} |{language} {_cell(r['text'])} |")
    return [*lines, ""]


def _variance(axes: card.Themes, i: int) -> str:
    """The axis' share of variance, and its language R² where languages vary."""
    share = f"{_share(axes.explained[i - 1])} of the variance"
    if not axes.corpus.mixed_languages:
        return f"{share}."
    r2 = axes.language_r2[i - 1]
    warning = " **Warning: mostly language.**" if r2 > axes.warning else ""
    return f"{share}; language R² {r2:.2f}.{warning}"


def _axis(ranked: pl.DataFrame, axes: card.Themes, i: int) -> list[str]:
    """Axis ``i``: its name, its variance, then the texts at each end (``ranked``: that axis' ends)."""
    languages = axes.corpus.mixed_languages
    low, high = axes.corpus.axes[i]
    name = f": {low} ↔ {high}" if low else ""
    return [
        f"## PC{i}{name}",
        "",
        _variance(axes, i),
        "",
        *_end(ranked.filter(end="high"), f"Highest ({high})" if high else "Highest", languages),
        *_end(ranked.filter(end="low"), f"Lowest ({low})" if low else "Lowest", languages),
    ]


def axes_markdown(ranked: pl.DataFrame, axes: card.Themes) -> str:
    """A corpus' axes report: each axis with its most extreme texts (and their languages)."""
    corpus = axes.corpus
    check = (
        " Language R²: share of the axis' variance across cells explained by the cell's "
        "most frequent language."
        if corpus.mixed_languages
        else ""
    )
    lines = [
        f"# What places talk about: the PCA axes of the {corpus.what}",
        "",
        f"The {axes.texts:,} {corpus.what} each count {corpus.placement} (resolution {geo.CELL_RESOLUTION}); "
        f"each cell takes the mean embedding of its texts, and each of the {axes.hexagons:,} resolution-"
        f"{geo.DISPLAY_RESOLUTION} cells with {corpus.min_texts} {corpus.unit} or more the mean of its child "
        "cells. "
        f"PCA of those means; scores below are the texts' projections on each axis.{check}",
        "",
    ]
    for i in range(1, COMPONENTS + 1):
        lines += _axis(ranked.filter(pc=i), axes, i)
    return "\n".join(lines)


def _layers(corpus: card.Corpus) -> str:
    return (
        f"Mean embedding of the {corpus.what} per H3 cell (resolution {geo.CELL_RESOLUTION}), "
        f"mean per resolution-{geo.DISPLAY_RESOLUTION} cell"
    )


def _grey(minimum: int, corpus: card.Corpus) -> str:
    return f"grey: land with fewer than {minimum} {corpus.unit} per resolution-{geo.DISPLAY_RESOLUTION} cell"


def render(work: Path, assets: Path, corpus: card.Corpus = LANDUSE) -> card.Themes:
    """A corpus: maps of the display cells (RGB, then one per axis) and its axes report in ``work``."""
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
        language_r2=stats.r_squared(scores, hexagons["language"].to_list()).tolist(),
        warning=LANGUAGE_WARNING,
    )
    land = geo.land_cells()
    grey = _grey(corpus.min_texts, corpus)
    caption = (
        f"{_layers(corpus)}: principal components 1, 2, 3 as red, green, blue "
        f"(each from its 2nd to 98th percentile)\n{grey}"
    )
    figures.rgb_map(index, stats.unit_scale(scores), assets / f"map_pca{corpus.suffix}.png", land, caption)
    for i in range(COMPONENTS):
        low, high = corpus.axes[i + 1]
        label = (
            f"PC{i + 1} ({_share(axes.explained[i])} of variance): {low or 'low'} ↔ {high or 'high'} ({grey})"
        )
        values = scores[:, i]
        path = assets / f"map_pc{i + 1}{corpus.suffix}.png"
        figures.diverging_map(index, values, stats.symmetric_limit(values), label, path, land)
    (work / corpus.report).write_text(axes_markdown(pl.read_parquet(out / "ends.parquet"), axes))
    return axes
