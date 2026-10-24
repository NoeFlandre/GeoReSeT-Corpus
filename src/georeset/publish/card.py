"""Render the dataset card (README.md) from computed stats."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from string import Template

import polars as pl

from georeset.normalize.languages import UNDETERMINED


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(x: float) -> str:
    return f"{x:.1%}"


Column = tuple[str, Callable]


def _line(cells: list[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _align(formatter: Callable) -> str:
    return "---" if formatter is str else "---:"


def table(df: pl.DataFrame, columns: dict[str, Column]) -> str:
    """Markdown table; text columns left-aligned, the rest right-aligned."""
    specs = list(columns.values())
    header = [_line([head for head, _ in specs]), f"|{'|'.join(_align(f) for _, f in specs)}|"]
    rows = (
        _line([f(v) for (_, f), v in zip(specs, row, strict=True)])
        for row in df.select(list(columns)).iter_rows()
    )
    return "\n".join([*header, *rows])


def _config(name: str, path: str) -> dict:
    return {"config_name": name, "data_files": [{"split": "train", "path": path}]}


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
class Context:
    repo_id: str
    by_source: pl.DataFrame
    continents: pl.DataFrame
    unlabelled: pl.DataFrame
    partitions: list[tuple[str, str]]
    polygons: int
    sources: dict[str, str]
    code_url: str


def _bold_total(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(column) == "total").then(pl.lit("**total**")).otherwise(pl.col(column)).alias(column)
    )


def _total(ctx: Context) -> dict:
    return ctx.by_source.filter(pl.col("source") == "total").row(0, named=True)  # pragma: no mutate (one row)


def _front_matter(ctx: Context) -> str:
    meta = {
        "license": ["odbl", "cc-by-sa-4.0", "other"],
        "pretty_name": "GeoReSeT Corpus",
        "language": iso_codes([lang for _, lang in ctx.partitions]),
        "task_categories": ["text-classification"],
        "tags": ["openstreetmap", "land-use", "land-cover", "geospatial", "multilingual"],
        "size_categories": [size_category(_total(ctx)["sentences"])],
        "configs": configs(ctx.partitions),
    }
    # JSON is valid YAML: no YAML dependency, no quoting surprises.
    return "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items())


def render(ctx: Context) -> str:
    total = _total(ctx)
    template = Template(files("georeset.publish").joinpath("templates/card.md").read_text())
    languages = sorted({lang for _, lang in ctx.partitions})
    return template.substitute(
        front_matter=_front_matter(ctx),
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
        unlabelled=table(
            _bold_total(ctx.unlabelled, "source"),
            {"source": ("Source", str), "failed": ("`failed`", fmt), "unsplit": ("`unsplit`", fmt)},
        ),
        sources="\n".join(
            f"   * [`{repo}` @ `{rev[:7]}`](https://huggingface.co/datasets/{repo}/tree/{rev})"
            for repo, rev in ctx.sources.items()
        ),
        code_url=ctx.code_url,
    )
