"""Compute the card's stats and figures from the published files, then write README.md."""

import json
import shutil
from importlib.resources import files
from pathlib import Path

import polars as pl

from georeset.build.pipeline import Layout
from georeset.publish import card, figures, geo, stats

MAP_RESOLUTION = 3
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


def build_card(layout: Layout, repo_id: str, code_url: str) -> Path:
    data = pl.scan_parquet(layout.out / "data" / "*" / "*" / "*.parquet")
    land = geo.load_land(layout.work / "cache")
    figures.world_map(
        geo.hex_counts(stats.points(data), MAP_RESOLUTION), land, layout.out / "assets" / "map.png"
    )
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
    )
    hero = HERO if context.scenes is None else FUSED_HERO
    shutil.copyfile(str(hero), layout.out / "assets" / "hero.png")
    readme = layout.out / "README.md"
    readme.write_text(card.render(context))
    return readme
