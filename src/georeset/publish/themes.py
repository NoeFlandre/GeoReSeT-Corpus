"""What places talk about: PCA of the mean text embedding per 2° hexagon, one per corpus.

Two corpora: the relevant sentences (``LANDUSE``) and the scene descriptions (``SCENES``).
``write_sentences`` splits a corpus' texts into its shards; ``scripts/embed.py`` embeds each into
the same-named file of the corpus' folder (``id``, ``lat``, ``lon``, ``vector``), row for row.
``compute`` reads one pair of shards at a time, twice: hexagon means first, then sentence scores.
``render`` turns its outputs into the maps and ``pca_axes.md``.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from georeset.normalize.languages import display_name
from georeset.publish import card, figures, geo, stats

COMPONENTS = 3
EXTREMES = 10  # sentences listed per end of an axis
MIX = 1_000  # sentences per end whose languages are counted
MIX_TOP = 5
LANGUAGE_WARNING = 0.5
MODEL = "microsoft/harrier-oss-v1-0.6b"  # as in scripts/embed.py
PROMPT = "Instruct: Identify the land use or land cover described\nQuery: "
PCS = [f"pc{i}" for i in range(1, COMPONENTS + 1)]
LANDUSE = card.Corpus(
    name="landuse",
    source="data/*/*/*.parquet",
    folder="embeddings",
    what="relevant sentences",
    unit="sentences",
    selection="distinct sentences with `landuse` true (the first `id` keeps a repeated text)",
    shards=128,
    min_texts=50,
    axes={1: ("", ""), 2: ("", ""), 3: ("", "")},
    languages=True,
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
        1: ("dense towns", "arid open land"),
        2: ("paved roads", "lakes and forests"),
        3: ("roads through forest", "historic city centres"),
    },
    languages=False,
    suffix="_scenes",
)
CORPORA = {corpus.name: corpus for corpus in (LANDUSE, SCENES)}
_SELECT = {LANDUSE.name: stats.landuse_sentences, SCENES.name: stats.scene_sentences}


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
    return pl.Series("id", table.column("id").to_pylist()), vectors.astype(np.float32)


def _pairs(sentences: list[Path], embeddings: list, filesystem) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    """Each sentence shard with a ``hexagon`` column, and the vectors of its rows."""
    for text, source in zip(sentences, embeddings, strict=True):
        rows = pl.read_parquet(text)
        ids, vectors = read_vectors(source, filesystem)
        if not rows["id"].equals(ids, check_names=False):
            raise ValueError(f"{source} does not match {text} row for row")
        yield rows.with_columns(hexagon=geo.hexagons(rows["lon"].to_numpy(), rows["lat"].to_numpy())), vectors


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


def _hexagon_totals(pairs) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """First pass: vector sums, sentence counts and (hexagon, language) counts."""
    sums, counts, languages = None, np.zeros(geo.HEXAGONS, dtype=np.int64), []
    for rows, vectors in pairs:
        shard_sums, shard_counts = stats.hexagon_sums(rows["hexagon"].to_numpy(), vectors)
        sums = shard_sums if sums is None else sums + shard_sums
        counts += shard_counts
        languages.append(rows.group_by("hexagon", "language").agg(n=pl.len()))
    mix = pl.concat(languages).group_by("hexagon", "language").agg(pl.col("n").sum())
    return np.asarray(sums), counts, mix


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
    sentences: list[Path], embeddings: list, work: Path, corpus: card.Corpus = LANDUSE, filesystem=None
) -> Path:
    """Hexagon means, their PCA, and each axis' most extreme texts, into ``work/themes/<corpus>``."""
    sums, counts, mix = _hexagon_totals(_pairs(sentences, embeddings, filesystem))
    kept = np.flatnonzero(counts >= corpus.min_texts)
    means = sums[kept] / counts[kept, None]
    fit = stats.pca(means, COMPONENTS)
    hexagons = _with_scores(pl.DataFrame({"hexagon": kept, "n": counts[kept]}), fit.scores(means))
    best = [
        ends(_with_scores(rows, fit.scores(vectors)), MIX)
        for rows, vectors in _pairs(sentences, embeddings, filesystem)
    ]
    out = work / "themes" / corpus.name
    out.mkdir(parents=True, exist_ok=True)
    hexagons.join(stats.dominant_language(mix), on="hexagon").write_parquet(out / "hexagons.parquet")
    _final_ends(pl.concat(best)).write_parquet(out / "ends.parquet")
    summary = {"sentences": int(counts.sum()), "explained": fit.explained.tolist()}
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
    listed = rows.filter(pl.col("rank") <= EXTREMES)
    if not languages:
        return [
            f"### {title}",
            "",
            "| # | Score | Text |",
            "|---:|---:|---|",
            *(
                f"| {r['rank']} | {r['score']:.3f} | {_cell(r['text'])} |"
                for r in listed.iter_rows(named=True)
            ),
            "",
        ]
    return [
        f"### {title}",
        "",
        f"Languages of the {rows.height:,} sentences at this end: {_mix(rows)}.",
        "",
        "| # | Score | Language | Sentence |",
        "|---:|---:|---|---|",
        *(
            f"| {r['rank']} | {r['score']:.3f} | {r['language']} | {_cell(r['text'])} |"
            for r in listed.iter_rows(named=True)
        ),
        "",
    ]


def _variance(axes: card.Themes, i: int) -> str:
    """The axis' share of variance, and its language R² where languages vary."""
    share = f"{_share(axes.explained[i - 1])} of the variance"
    if not axes.corpus.languages:
        return f"{share}."
    r2 = axes.language_r2[i - 1]
    warning = " **Warning: mostly language.**" if r2 > axes.warning else ""
    return f"{share}; language R² {r2:.2f}.{warning}"


def _axis(ends: pl.DataFrame, axes: card.Themes, i: int) -> list[str]:
    languages = axes.corpus.languages
    low, high = axes.corpus.axes[i]
    name = f": {low} ↔ {high}" if low else ""
    return [
        f"## PC{i}{name}",
        "",
        _variance(axes, i),
        "",
        *_end(ends.filter(end="high"), f"Highest ({high})" if high else "Highest", languages),
        *_end(ends.filter(end="low"), f"Lowest ({low})" if low else "Lowest", languages),
    ]


def axes_markdown(ends: pl.DataFrame, axes: card.Themes) -> str:
    """A corpus' axes report: each axis with its most extreme texts (and their languages)."""
    corpus = axes.corpus
    check = (
        " Language R²: share of the axis' variance across hexagons explained by the hexagon's "
        "most frequent language."
        if corpus.languages
        else ""
    )
    lines = [
        f"# What places talk about: the PCA axes of the {corpus.what}",
        "",
        f"PCA of the mean embedding of the {axes.texts:,} {corpus.what} in each of "
        f"{axes.hexagons:,} {geo.HEX_DEGREES}° hexagons with {corpus.min_texts} {corpus.unit} or more. "
        f"Scores below are projections on each axis.{check}",
        "",
    ]
    for i in range(1, COMPONENTS + 1):
        lines += _axis(ends.filter(pc=i), axes, i)
    return "\n".join(lines)


def render(work: Path, assets: Path, corpus: card.Corpus = LANDUSE) -> card.Themes:
    """A corpus' maps of the hexagon scores (RGB, then one per component) and its axes report in ``work``."""
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
    figures.rgb_map(index, stats.unit_scale(scores), assets / f"map_pca{corpus.suffix}.png")
    for i in range(COMPONENTS):
        low, high = corpus.axes[i + 1]
        label = f"PC{i + 1} ({_share(axes.explained[i])} of variance): {low or 'low'} ↔ {high or 'high'}"
        values = scores[:, i]
        figures.diverging_map(
            index, values, stats.symmetric_limit(values), label, assets / f"map_pc{i + 1}{corpus.suffix}.png"
        )
    (work / corpus.report).write_text(axes_markdown(pl.read_parquet(out / "ends.parquet"), axes))
    return axes
