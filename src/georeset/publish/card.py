"""Render the dataset card (README.md) from computed stats."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from string import Template

import polars as pl

from georeset.normalize.languages import UNDETERMINED
from georeset.sources import scenes as scenes_source

TEMPLATE = files("georeset.publish").joinpath("templates/card.md")
SCENES_TEMPLATE = files("georeset.publish").joinpath("templates/scenes.md")
SCENE_PARTS = ("headline", "intro", "quickstart", "files", "configs", "section", "method", "license")
HERO_ALT = (
    "GeoReSeT: collect texts linked to OpenStreetMap polygons, filter them by land-use relevance, "
    "link sentences back to polygons"
)
FUSED_HERO_ALT = (
    "GeoReSeT Fused: texts linked to OpenStreetMap polygons are filtered by land-use relevance; "
    "geotagged photos are clustered and described by a vision-language model; "
    "both are tied back to places"
)
_PART = re.compile(r"^<!-- (\w+) -->\n", re.MULTILINE)


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(x: float) -> str:
    return f"{x:.1%}"


def area(m2: float) -> str:
    """Readable area: m² below a hectare, hectares below a km², then km²."""
    if m2 < 10_000:
        return f"{m2:,.0f} m²"
    if m2 < 1_000_000:
        return f"{m2 / 10_000:,.1f} ha"
    return f"{m2 / 1_000_000:,.1f} km²"


Column = tuple[str, Callable]


def _line(cells: list[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _align(formatter: Callable) -> str:
    return "---" if formatter is str else "---:"


def table(df: pl.DataFrame, columns: dict[str, Column]) -> str:
    """Markdown table; text columns left-aligned, the rest right-aligned."""
    specs = list(columns.values())
    header = [_line([head for head, _ in specs]), f"|{'|'.join(_align(f) for _, f in specs)}|"]
    rows = (_line([f(row[c]) for c, (_, f) in columns.items()]) for row in df.iter_rows(named=True))
    return "\n".join([*header, *rows])


def _config(name: str, path: str) -> dict:
    return {"config_name": name, "data_files": [{"split": "train", "path": path}]}


SCENE_CONFIGS = ("scenes", "scene_photos")


def configs(partitions: list[tuple[str, str]]) -> list[dict]:
    """``all`` (default), one per source, one per language, then ``unlabelled`` and ``polygons``."""
    sources = sorted({s for s, _ in partitions})
    languages = sorted({lang for _, lang in partitions})
    return [
        {**_config("all", "data/*/*/*.parquet"), "default": True},
        *(_config(s, f"data/{s}/*/*.parquet") for s in sources),
        *(_config(lang, f"data/*/{lang}/*.parquet") for lang in languages),
        _config("unlabelled", "unlabelled/*.parquet"),
        _config("polygons", "polygons/*.parquet"),
    ]


def folder_configs(names: tuple[str, ...]) -> list[dict]:
    """One config per top-level folder."""
    return [_config(name, f"{name}/*.parquet") for name in names]


_SIZES = [
    (10**3, "n<1K"),
    (10**4, "1K<n<10K"),
    (10**5, "10K<n<100K"),
    (10**6, "100K<n<1M"),
    (10**7, "1M<n<10M"),
    (10**8, "10M<n<100M"),
    (10**9, "100M<n<1B"),
]


def size_category(n: int) -> str:
    """Hugging Face ``size_categories`` bucket of ``n`` rows."""
    return next((label for bound, label in _SIZES if n < bound), "1B<n<10B")


def iso_codes(languages: list[str]) -> list[str]:
    return sorted({lang.split("_")[0] for lang in languages if lang != UNDETERMINED})


@dataclass(frozen=True)
class Scenes:
    """The geo-rag scene descriptions fused into the corpus."""

    totals: dict[str, int]
    labels: pl.DataFrame
    platforms: pl.DataFrame
    locality: pl.DataFrame
    raw: dict[str, str]


@dataclass(frozen=True)
class Context:
    repo_id: str
    by_source: pl.DataFrame
    continents: pl.DataFrame
    unlabelled: pl.DataFrame
    partitions: list[tuple[str, str]]
    polygons: int
    areas: pl.DataFrame
    sources: dict[str, str]
    code_url: str
    scenes: Scenes | None = None


def _bold_total(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(column) == "total").then(pl.lit("**total**")).otherwise(pl.col(column)).alias(column)
    )


def _total(ctx: Context) -> dict:
    return ctx.by_source.filter(pl.col("source") == "total").row(0, named=True)  # pragma: no mutate (one row)


@dataclass(frozen=True)
class Flavor:
    """What differs between the plain corpus card and the fused one."""

    title: str
    hero_alt: str
    licenses: tuple[str, ...]
    tags: tuple[str, ...]
    folders: tuple[str, ...]


_TAGS = ("openstreetmap", "land-use", "land-cover", "geospatial", "multilingual")
PLAIN = Flavor("GeoReSeT Corpus", HERO_ALT, ("odbl", "cc-by-sa-4.0", "other"), _TAGS, ())
FUSED = Flavor(
    "GeoReSeT Corpus Fused",
    FUSED_HERO_ALT,
    ("odbl", "cc-by-sa-4.0", "cc-by-4.0", "other"),
    (*_TAGS, "street-level-imagery", "scene-description"),
    SCENE_CONFIGS,
)


def flavor(ctx: Context) -> Flavor:
    return PLAIN if ctx.scenes is None else FUSED


def _front_matter(ctx: Context) -> str:
    kind = flavor(ctx)
    meta = {
        "license": list(kind.licenses),
        "pretty_name": kind.title,
        "language": iso_codes([lang for _, lang in ctx.partitions]),
        "task_categories": ["text-classification"],
        "tags": list(kind.tags),
        "size_categories": [size_category(_total(ctx)["sentences"])],
        "configs": configs(ctx.partitions) + folder_configs(kind.folders),
    }
    # JSON is valid YAML: no YAML dependency, no quoting surprises.
    return "\n".join(f"{key}: {json.dumps(value)}" for key, value in meta.items())


def pinned(repo: str, rev: str) -> str:
    """Markdown link to a dataset repo at one commit."""
    return f"[`{repo}` @ `{rev[:7]}`](https://huggingface.co/datasets/{repo}/tree/{rev})"


def km(n: int) -> str:
    return f"{n:,} km"


def parts(text: str) -> dict[str, str]:
    """Named parts of a template split on ``<!-- name -->`` lines, without their final newline."""
    pieces = _PART.split(text)
    pairs = zip(pieces[1::2], pieces[2::2], strict=True)  # pragma: no mutate (split pairs are always even)
    return {name: body.removesuffix("\n") for name, body in pairs}


def _scene_values(ctx: Context, scenes: Scenes) -> dict[str, str]:
    totals = scenes.totals
    ((raw_repo, raw_rev),) = scenes.raw.items()
    return {
        "repo_id": ctx.repo_id,
        "scene_texts": fmt(totals["scenes"]),
        "scene_photos": fmt(totals["photos"]),
        "scene_words": fmt(totals["words"]),
        "scene_countries": fmt(totals["countries"]),
        "scene_median": fmt(totals["median_photos"]),
        "scene_locality": table(scenes.locality, {f"km{d}": (km(d), fmt) for d in (10, 100, 1_000)}),
        "scene_labels": table(
            scenes.labels,
            {"lulc_label": ("Label", str), "scenes": ("Scenes", fmt), "photos": ("Photos", fmt)},
        ),
        "scene_platforms": table(
            scenes.platforms,
            {"platform": ("Platform", str), "photos": ("Photos", fmt), "linked": ("With `url`", pct)},
        ),
        "scene_raw": f"{pinned(raw_repo, raw_rev)} (private)",
        "geo_rag_url": scenes_source.CODE_URL,
        "geo_rag_rev": scenes_source.CODE_REV,
        "geo_rag_short": scenes_source.CODE_REV[:7],
    }


def scene_parts(ctx: Context) -> dict[str, str]:
    """The fused card's extra text, or empty strings for the plain corpus."""
    if ctx.scenes is None:
        return {name: "" for name in SCENE_PARTS}
    values = _scene_values(ctx, ctx.scenes)
    return {
        name: Template(body).substitute(values) for name, body in parts(SCENES_TEMPLATE.read_text()).items()
    }


def render(ctx: Context) -> str:
    total = _total(ctx)
    template = Template(TEMPLATE.read_text())
    languages = sorted({lang for _, lang in ctx.partitions})
    kind = flavor(ctx)
    return template.substitute(
        {f"scenes_{name}": text for name, text in scene_parts(ctx).items()},
        front_matter=_front_matter(ctx),
        title=kind.title,
        hero_alt=kind.hero_alt,
        repo_id=ctx.repo_id,
        sentences=fmt(total["sentences"]),
        polygons=fmt(ctx.polygons),
        languages=fmt(len(languages)),
        regions=fmt(total["regions"]),
        example_language="fra_Latn" if "fra_Latn" in languages else languages[0],
        by_source=table(
            _bold_total(ctx.by_source, "source"),
            {
                "source": ("Source", str),
                "sentences": ("Sentences", fmt),
                "words": ("Words", fmt),
                "polygons": ("Polygons", fmt),
                "languages": ("Languages", fmt),
                "regions": ("Regions", fmt),
                "landuse": ("`landuse` share", pct),
            },
        ),
        continents=table(
            ctx.continents,
            {"continent": ("Continent", str), "sentences": ("Sentences", fmt), "polygons": ("Polygons", fmt)},
        ),
        areas=table(
            ctx.areas,
            {
                "p10": ("10%", area),
                "p25": ("25%", area),
                "p50": ("Median", area),
                "p75": ("75%", area),
                "p90": ("90%", area),
            },
        ),
        unlabelled=table(
            _bold_total(ctx.unlabelled, "source"),
            {"source": ("Source", str), "failed": ("`failed`", fmt), "unsplit": ("`unsplit`", fmt)},
        ),
        sources="\n".join(f"   * {pinned(repo, rev)}" for repo, rev in ctx.sources.items()),
        code_url=ctx.code_url,
    )
