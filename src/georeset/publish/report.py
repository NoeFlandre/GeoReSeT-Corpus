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


def partitions(out: Path) -> list[tuple[str, str]]:
    return sorted((d.parent.name, d.name) for d in out.glob("data/*/*"))


def build_card(layout: Layout, repo_id: str, code_url: str) -> Path:
    data = pl.scan_parquet(layout.out / "data" / "*" / "*" / "*.parquet")
    land = geo.load_land(layout.work / "cache")
    figures.world_map(
        geo.hex_counts(stats.points(data), MAP_RESOLUTION), land, layout.out / "assets" / "map.png"
    )
    figures.language_bars(stats.top_languages(data, TOP_LANGUAGES), layout.out / "assets" / "languages.png")
    context = card.Context(
        repo_id=repo_id,
        by_source=stats.by_source(data),
        continents=stats.by_continent(data),
        unlabelled=stats.unlabelled(pl.scan_parquet(layout.out / "unlabelled" / "*.parquet")),
        partitions=partitions(layout.out),
        polygons=pl.scan_parquet(layout.out / "polygons" / "*.parquet").select(pl.len()).collect().item(),
        sources=json.loads((layout.work / "revisions.json").read_text()),
        code_url=code_url,
    )
    hero = files("georeset.publish").joinpath("assets/hero.png")
    shutil.copyfile(str(hero), layout.out / "assets" / "hero.png")
    readme = layout.out / "README.md"
    readme.write_text(card.render(context))
    return readme
