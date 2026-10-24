"""``georeset`` command line: each step is resumable, ``run`` chains them all."""

from pathlib import Path
from typing import Annotated

import typer

from georeset import hub, pipeline, report, stage
from georeset.sources import description, website, wiki

SOURCES = (website.SOURCE, description.SOURCE, wiki.SOURCE)
REPO_ID = "NoeFlandre/GeoReSeT-Corpus"
CODE_URL = "https://github.com/NoeFlandre/GeoReSeT-Corpus"

app = typer.Typer(no_args_is_help=True, add_completion=False)
Work = Annotated[Path, typer.Option(help="Scratch directory for staged data.")]
Out = Annotated[Path, typer.Option(help="Directory holding the publishable dataset.")]
Workers = Annotated[int, typer.Option(help="Parallel memory-heavy jobs.")]
Downloads = Annotated[int, typer.Option(help="Parallel downloads.")]
Regions = Annotated[list[str] | None, typer.Option("--region", help="Geofabrik stems; default: all.")]


def _stems(regions: list[str] | None) -> list[str]:
    if regions:
        return regions
    available = stage.hub.list_files(wiki.SOURCE.repo, stage.hub.revision(wiki.SOURCE.repo))
    return sorted(Path(p).stem for p in available if p.startswith("polygons/"))


@app.command("stage")
def stage_cmd(
    work: Work = Path("work"), workers: Workers = 3, downloads: Downloads = 12, region: Regions = None
) -> None:
    """Download and convert every (source, region) to staged tables."""
    stage.stage(stage.tasks(SOURCES, _stems(region)), work, downloads, workers)


@app.command()
def meta(work: Work = Path("work"), out: Out = Path("out")) -> None:
    """Pick one row per polygon and one copy per document."""
    pipeline.build_meta(pipeline.Layout(work, out))


@app.command()
def rows(work: Work = Path("work"), out: Out = Path("out"), workers: Workers = 3) -> None:
    """Build final rows for every staged region."""
    layout = pipeline.Layout(work, out)
    jobs = [lambda s=s: pipeline.build_rows(layout, s) for s in sorted(layout.stage.glob("*/*"))]
    pipeline.run_parallel(jobs, workers)


@app.command()
def shards(work: Work = Path("work"), out: Out = Path("out"), workers: Workers = 3) -> None:
    """Sort and write the final parquet shards."""
    pipeline.build_shards(pipeline.Layout(work, out), workers)


@app.command("card")
def card_cmd(work: Work = Path("work"), out: Out = Path("out"), repo: str = REPO_ID) -> None:
    """Compute stats and figures, write README.md."""
    typer.echo(report.build_card(pipeline.Layout(work, out), repo, CODE_URL))


@app.command()
def publish(out: Out = Path("out"), repo: str = REPO_ID) -> None:
    """Upload the dataset to the Hugging Face Hub (public)."""
    typer.echo(hub.publish(out, repo))


@app.command()
def run(
    work: Work = Path("work"),
    out: Out = Path("out"),
    workers: Workers = 3,
    downloads: Downloads = 12,
    region: Regions = None,
    publish_: Annotated[bool, typer.Option("--publish", help="Upload when done.")] = False,
) -> None:
    """Every step, from download to the dataset card, then optionally publish."""
    stage_cmd(work, workers, downloads, region)
    meta(work, out)
    rows(work, out, workers)
    shards(work, out, workers)
    card_cmd(work, out)
    if publish_:
        publish(out)
