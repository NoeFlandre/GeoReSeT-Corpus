"""Compute the card's stats and figures from the published files, then write README.md."""

import json
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from georeset.build.pipeline import Layout
from georeset.publish import card, figures, geo, stats, themes

# Sentence maps after the full dataset (map.png): file name -> rows drawn.
SENTENCE_MAPS = {
    "map_landuse.png": pl.col("landuse"),
    "map_website.png": pl.col("source") == "website",
    "map_description.png": pl.col("source") == "osm_description",
    "map_wikimedia.png": pl.col("source").is_in(["wikipedia", "wikivoyage"]),
}
TOP_LANGUAGES = 20
HERO = files("georeset.publish").joinpath("assets/hero.png")
FUSED_HERO = files("georeset.publish").joinpath("assets/hero_fused.png")
TOP_LABELS = 10


def partitions(out: Path) -> list[tuple[str, str]]:
    return sorted((d.parent.name, d.name) for d in out.glob("data/*/*"))


def scene_context(out: Path, work: Path) -> card.Scenes | None:
    """Stats of the fused scenes, or ``None`` when ``out`` holds the plain corpus."""
    if not (out / "scenes").is_dir():
        return None
    scenes = pl.scan_parquet(out / "scenes" / "*.parquet")
    return card.Scenes(
        totals=stats.scene_totals(scenes),
        labels=stats.scene_labels(scenes, TOP_LABELS),
        platforms=stats.scene_platforms(pl.scan_parquet(out / "scene_photos" / "*.parquet")),
        locality=stats.scene_locality(scenes),
        raw=json.loads((work / "scenes.json").read_text()),
    )


def scene_points(out: Path) -> pl.DataFrame | None:
    """Photos per location of the fused scenes, or ``None`` for the plain corpus."""
    if not (out / "scene_photos").is_dir():
        return None
    return stats.points(pl.scan_parquet(out / "scene_photos" / "*.parquet"))


def display_counts(points: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each display cell and the mean count of its child H3 cells (each cell counts once)."""
    counts = geo.hex_counts(points)
    parents, sums, sizes, _ = stats.parent_sums(counts["cell"].to_numpy(), counts["n"].to_numpy()[:, None])
    return parents, sums[:, 0] / sizes


def draw_maps(data: pl.LazyFrame, out: Path) -> None:
    """The full dataset map, then the sentence subsets and the scenes over it in grey."""
    maps = out / "assets"
    photos = scene_points(out)
    sentences = stats.points(data)
    full, values = display_counts(sentences if photos is None else pl.concat([sentences, photos]))
    unit = "Sentences" if photos is None else "Sentences and scene photos"
    figures.world_map(full, values, maps / "map.png", unit)
    for name, rows in SENTENCE_MAPS.items():
        figures.world_map(*display_counts(stats.points(data.filter(rows))), maps / name, "Sentences", full)
    if photos is not None:
        figures.world_map(*display_counts(photos), maps / "map_scenes.png", "Photos", full)


def theme_context(work: Path, assets: Path) -> tuple[card.Themes, ...]:
    """Maps of each embeddings PCA that ``georeset themes`` has computed."""
    return tuple(
        themes.render(work, assets, corpus)
        for corpus in themes.CORPORA.values()
        if (work / "themes" / corpus.name / "summary.json").is_file()
    )


TRANSLATION_COLUMNS = ["translation", "translation_language", "translation_flags"]


def translation_context(out: Path, work: Path) -> card.Translation | None:
    """Stats of the English translations (``work/translation.json`` and the shards), if they were added."""
    summary = work / "translation.json"
    if not summary.is_file():
        return None
    folders = (out / "data" / "*" / "*" / "*.parquet", out / "unlabelled" / "*.parquet")
    rows = pl.concat([pl.scan_parquet(f).select(TRANSLATION_COLUMNS) for f in folders])
    meta = json.loads(summary.read_text())
    return card.Translation(
        kinds=stats.translation_kinds(rows),
        flags=stats.translation_flags(rows),
        distinct=meta["distinct"],
        languages=stats.translation_languages(rows),
        model=meta["model"],
        revision=meta["revision"],
        threshold=meta["threshold"],
    )


def _rows(lazy: pl.LazyFrame) -> int:
    return lazy.select(pl.len()).collect().item()


def extras_context(out: Path, work: Path) -> card.Extras | None:
    """Counts of the folders copied from the fused dataset (pinned in ``work/fused.json``), if they were."""
    pins = work / "fused.json"
    if not pins.is_file():
        return None
    shares = pl.scan_parquet(out / "polygons_labels_shares" / "*.parquet")
    return card.Extras(
        labelled=_rows(shares.filter(pl.col("max_eunis_share_code").is_not_null())),
        fused=json.loads(pins.read_text()),
    )


def english_context(out: Path) -> card.English | None:
    """Counts of the English translations' vectors (``embeddings_en``), if they were added."""
    folder = out / "embeddings_en"
    if not folder.is_dir():
        return None
    return card.English(
        embedded=_rows(pl.scan_parquet(folder / "*.parquet")),
        model=themes.MODEL,
        prompt=themes.PROMPT,
        tokens=themes.MAX_TOKENS,
    )


def build_card(layout: Layout, repo_id: str, code_url: str) -> Path:
    data = pl.scan_parquet(layout.out / "data" / "*" / "*" / "*.parquet")
    draw_maps(data, layout.out)
    figures.language_bars(stats.top_languages(data, TOP_LANGUAGES), layout.out / "assets" / "languages.png")
    polygons = pl.scan_parquet(layout.out / "polygons" / "*.parquet")
    context = card.Context(
        repo_id=repo_id,
        by_source=stats.by_source(data),
        continents=stats.by_continent(data),
        unlabelled=stats.unlabelled(pl.scan_parquet(layout.out / "unlabelled" / "*.parquet")),
        partitions=partitions(layout.out),
        polygons=polygons.select(pl.len()).collect().item(),
        areas=stats.polygon_areas(polygons),
        sources=json.loads((layout.work / "revisions.json").read_text()),
        code_url=code_url,
        scenes=scene_context(layout.out, layout.work),
        themes=theme_context(layout.work, layout.out / "assets"),
        translation=translation_context(layout.out, layout.work),
        extras=extras_context(layout.out, layout.work),
        english=english_context(layout.out),
    )
    hero = HERO if context.scenes is None else FUSED_HERO
    shutil.copyfile(str(hero), layout.out / "assets" / "hero.png")
    readme = layout.out / "README.md"
    readme.write_text(card.render(context))
    return readme
