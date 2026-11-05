"""What places talk about: PCA of the mean text embedding per 2° hexagon, one per corpus.

Three corpora: the relevant sentences (``LANDUSE``), the same in English only (``ENGLISH``, on the same
vectors) and the scene descriptions (``SCENES``). ``write_sentences`` splits a corpus' texts into its
shards; ``scripts/embed.py`` embeds each into the same-named file of the corpus' folder (``id``, ``lat``,
``lon``, ``vector``), row for row. ``compute`` reads one pair of shards at a time, twice: hexagon means
and their PCA first, then each text's scores (axis ends, fine-hexagon means). ``render`` turns its
outputs into the maps and the corpus' axes report.
"""

import json
from collections.abc import Iterator
from dataclasses import replace
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
LANDUSE = card.Corpus(
    name="landuse",
    source="data/*/*/*.parquet",
    folder="embeddings",
    what="relevant sentences",
    unit="sentences",
    selection="distinct sentences with `landuse` true (the first `id` keeps a repeated text)",
    shards=128,
    min_texts=50,
    axes={
        1: ("tropical beaches and leisure", "Russian village histories"),
        2: ("buildings and transport hubs", "islands and rocky coasts"),
        3: ("remote airports", "church ornament and architecture"),
    },
    mixed_languages=True,
    placement="at its location",
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
        2: ("lakes and forests", "dry open land"),
        3: ("lush tropical vegetation", "snow, ice and rock"),
    },
    mixed_languages=False,
    placement="once in every hexagon where its scene has photos",
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
        1: ("wild coasts and wilderness", "historic city buildings"),
        2: ("ancient ruins and history", "park and community amenities"),
        3: ("tropical beaches and resorts", "glacial lakes and valleys"),
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


def _pairs(
    sentences: list[Path], embeddings: list, filesystem, language: str = ""
) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    """Each sentence shard and the vectors of its rows; only the rows in ``language`` if given."""
    for text, source in zip(sentences, embeddings, strict=True):
        rows = pl.read_parquet(text)
        ids, vectors = read_vectors(source, filesystem)
        if not rows["id"].equals(ids, check_names=False):
            raise ValueError(f"{source} does not match {text} row for row")
        if language:
            keep = (rows["language"] == language).to_numpy()
            rows, vectors = rows.filter(keep), vectors[keep]
        yield rows, vectors


def _place(rows: pl.DataFrame, places: pl.DataFrame | None, grid: geo.Grid) -> tuple[np.ndarray, np.ndarray]:
    """The ``grid`` hexagon each row counts in and that row's index: its location's, or each of ``places``."""
    if places is None:
        return grid.hexagons(rows["lon"].to_numpy(), rows["lat"].to_numpy()), np.arange(rows.height)
    placed = rows.select("id").with_row_index("row").join(places, on="id")
    return placed["hexagon"].to_numpy(), placed["row"].to_numpy()


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


def _hexagon_totals(pairs, places: pl.DataFrame | None) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """First pass: vector sums, text counts and (hexagon, language) counts."""
    sums, counts, languages = None, np.zeros(geo.COARSE.count, dtype=np.int64), []
    for rows, vectors in pairs:
        hexes, index = _place(rows, places, geo.COARSE)
        shard_sums, shard_counts = stats.hexagon_sums(hexes, vectors[index])
        sums = shard_sums if sums is None else sums + shard_sums
        counts += shard_counts
        placed = pl.DataFrame({"hexagon": hexes, "language": rows["language"].gather(index)})
        languages.append(placed.group_by("hexagon", "language").agg(n=pl.len()))
    mix = pl.concat(languages).group_by("hexagon", "language").agg(pl.col("n").sum())
    return np.asarray(sums), counts, mix


def _scored(
    pairs, fit: stats.Pca, places: pl.DataFrame | None, fine_min: int
) -> tuple[int, pl.DataFrame, pl.DataFrame]:
    """Second pass: the number of texts, each axis' most extreme texts, and mean scores per fine hexagon."""
    texts, best, fine = 0, [], np.zeros((geo.FINE.count, COMPONENTS + 1))
    for rows, vectors in pairs:
        scores = fit.scores(vectors)
        texts += rows.height
        best.append(ends(_with_scores(rows, scores), MIX))
        hexes, index = _place(rows, places, geo.FINE)
        fine += stats.score_sums(hexes, scores[index], geo.FINE.count)
    kept = np.flatnonzero(fine[:, -1] >= fine_min)
    means = fine[kept, :-1] / fine[kept, -1:]
    hexagons = _with_scores(pl.DataFrame({"hexagon": kept, "n": fine[kept, -1].astype(np.int64)}), means)
    return texts, _final_ends(pl.concat(best)), hexagons


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
    """Hexagon means, their PCA, axis ends and fine-hexagon scores, in ``work/themes/<corpus>``.

    ``photos`` (``scene_id``, ``lat``, ``lon``): each text counts once in every hexagon of its photos,
    instead of at its own location.
    """
    places = {
        grid: None if photos is None else stats.photo_hexagons(photos, grid)
        for grid in (geo.COARSE, geo.FINE)
    }
    pairs = _pairs(sentences, embeddings, filesystem, corpus.only_language)
    sums, counts, mix = _hexagon_totals(pairs, places[geo.COARSE])
    kept = np.flatnonzero(counts >= corpus.min_texts)
    means = sums[kept] / counts[kept, None]
    fit = stats.pca(means, COMPONENTS)
    hexagons = _with_scores(pl.DataFrame({"hexagon": kept, "n": counts[kept]}), fit.scores(means))
    pairs = _pairs(sentences, embeddings, filesystem, corpus.only_language)
    texts, best, fine = _scored(pairs, fit, places[geo.FINE], corpus.fine_min)
    out = work / "themes" / corpus.name
    out.mkdir(parents=True, exist_ok=True)
    hexagons.join(stats.dominant_language(mix), on="hexagon").write_parquet(out / "hexagons.parquet")
    best.write_parquet(out / "ends.parquet")
    fine.write_parquet(out / "fine.parquet")
    summary = {"sentences": texts, "explained": fit.explained.tolist()}
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
        " Language R²: share of the axis' variance across hexagons explained by the hexagon's "
        "most frequent language."
        if corpus.mixed_languages
        else ""
    )
    lines = [
        f"# What places talk about: the PCA axes of the {corpus.what}",
        "",
        f"PCA of the mean embedding of the {axes.texts:,} {corpus.what}, each counted {corpus.placement}, "
        f"in each of {axes.hexagons:,} {geo.HEX_DEGREES}° hexagons with {corpus.min_texts} {corpus.unit} "
        f"or more. Scores below are projections on each axis.{check}",
        "",
    ]
    for i in range(1, COMPONENTS + 1):
        lines += _axis(ranked.filter(pc=i), axes, i)
    return "\n".join(lines)


def _grey(minimum: int, corpus: card.Corpus) -> str:
    return f"grey: land with fewer than {minimum} {corpus.unit} per hexagon"


def _fine_map(out: Path, assets: Path, corpus: card.Corpus, limits: np.ndarray) -> None:
    """The RGB map again on the fine grid, coloured as the coarse one (same axes, same limits)."""
    fine = pl.read_parquet(out / "fine.parquet")
    degrees = geo.FINE.degrees
    caption = (
        f"Mean embedding of the {corpus.what} per {degrees}° hexagon on the {geo.HEX_DEGREES}° map's "
        f"principal components, coloured as that map\n{_grey(corpus.fine_min, corpus)}"
    )
    rgb = stats.unit_scale(fine.select(PCS).to_numpy(), limits)
    path = assets / f"map_pca_fine{corpus.suffix}.png"
    figures.rgb_map(fine["hexagon"].to_numpy(), rgb, path, geo.FINE.land(), caption, geo.FINE)


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
    land = geo.COARSE.land()
    grey = _grey(corpus.min_texts, corpus)
    caption = (
        f"Mean embedding of the {corpus.what} per {geo.HEX_DEGREES}° hexagon: principal components 1, 2, 3 "
        f"as red, green, blue (each from its 2nd to 98th percentile)\n{grey}"
    )
    figures.rgb_map(index, stats.unit_scale(scores), assets / f"map_pca{corpus.suffix}.png", land, caption)
    _fine_map(out, assets, corpus, stats.percentile_limits(scores))
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
