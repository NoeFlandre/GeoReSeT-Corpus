"""``georeset`` command line: each step is resumable, ``run`` chains them all."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from georeset import embedding, hub
from georeset.build import extras, pipeline, stage
from georeset.build import fuse as fusion
from georeset.publish import report, space, themes
from georeset.sources import description, website, wiki
from georeset.translate import jobs as translation
from georeset.translate.store import Bucket

SOURCES = (website.SOURCE, description.SOURCE, wiki.SOURCE)
REPO_ID = "NoeFlandre/GeoReSeT-Corpus"
FUSED_REPO_ID = "NoeFlandre/GeoReSeT-Corpus-Fused"
CORPUS_RELEASE = "v1.0"  # tag of REPO_ID fused with the scenes
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


Sentences = Annotated[Path, typer.Option(help="Directory of the sentence shards to embed.")]


class CorpusName(StrEnum):
    """Keys of ``themes.CORPORA``."""

    landuse = "landuse"
    english = "english"
    scenes = "scenes"


Corpus = Annotated[
    CorpusName,
    typer.Option(
        help="landuse: the relevant sentences; english: those in English; scenes: the scene descriptions."
    ),
]


@app.command("themes-sentences")
def themes_sentences(
    corpus: Corpus = CorpusName.landuse,
    sentences: Sentences = Path("sentences"),
    data: Annotated[
        str, typer.Option(help="Parquet glob of the texts (default: the corpus on the Hub).")
    ] = "",
) -> None:
    """Write a corpus' distinct texts in shards, for scripts/embed.py."""
    chosen = themes.CORPORA[corpus]
    source = data or f"hf://datasets/{FUSED_REPO_ID}/{chosen.source}"
    typer.echo(themes.write_sentences(pl.scan_parquet(source), sentences, chosen))


@app.command("themes")
def themes_cmd(
    work: Work = Path("work-fused"),
    sentences: Sentences = Path("sentences"),
    cache: Annotated[
        Path, typer.Option(help="Where the embeddings are downloaded (~45 GB, node-local).")
    ] = Path("/tmp/georeset"),
    repo: str = FUSED_REPO_ID,
    corpus: Corpus = CorpusName.landuse,
) -> None:
    """PCA of a corpus' embeddings per hexagon, from its folder on the Hub, into ``work/themes/<corpus>``."""
    chosen = themes.CORPORA[corpus]
    revision = hub.revision(repo)
    hub.snapshot(repo, revision, cache, [f"{chosen.folder}/*"])
    photos = None
    if chosen.photos:
        located = pl.scan_parquet(f"hf://datasets/{repo}@{revision}/{chosen.photos}")
        photos = located.select("scene_id", "lat", "lon").collect(engine="streaming")
    shards = sorted(sentences.glob("train-*.parquet"))
    vectors = [cache / chosen.folder / s.name for s in shards]
    typer.echo(themes.compute(shards, vectors, work, chosen, photos))


@app.command("space")
def space_cmd(
    work: Work = Path("work-fused"),
    cache: Annotated[Path, typer.Option(help="Where the dataset is downloaded (node-local).")] = Path(
        "/tmp/georeset"
    ),
    out: Annotated[Path, typer.Option(help="The Space's files.")] = Path("space"),
    repo: str = FUSED_REPO_ID,
    target: Annotated[str, typer.Option(help="The Space to upload to; empty: build only.")] = FUSED_REPO_ID,
) -> None:
    """Build the explorer Space from the dataset (current revision) and ``georeset themes``; upload it."""
    revision = hub.revision(repo)
    hub.snapshot(repo, revision, cache, ["data/*", "scene_photos/*", "scenes/*"])
    space.build(
        pl.scan_parquet(cache / "data" / "*" / "*" / "*.parquet"),
        pl.scan_parquet(cache / "scene_photos" / "*.parquet"),
        pl.scan_parquet(cache / "scenes" / "*.parquet"),
        work,
        out / "data",
        repo,
        revision,
        hub.folder_commits(repo, revision, space.FOLDERS),
    )
    space.copy_app(out)
    typer.echo(hub.publish_space(out, target) if target else out)


@app.command()
def publish(
    out: Out = Path("out"),
    repo: str = REPO_ID,
    private: Annotated[bool, typer.Option(help="Create the repo private.")] = False,
) -> None:
    """Upload the dataset to the Hugging Face Hub."""
    typer.echo(hub.publish(out, repo, private))


@app.command()
def fuse(
    work: Work = Path("work-fused"),
    out: Out = Path("out-fused"),
    corpus: Annotated[str, typer.Option(help="Corpus release (tag or commit) to fuse.")] = CORPUS_RELEASE,
    repo: str = FUSED_REPO_ID,
    publish_: Annotated[bool, typer.Option("--publish", help="Upload when done.")] = False,
    private: Annotated[bool, typer.Option(help="Create the repo private.")] = False,
) -> None:
    """Copy a corpus release, add the geo-rag scenes, write the card, then optionally publish."""
    fusion.fetch_corpus(REPO_ID, corpus, out, work)
    fusion.build_scenes(fusion.fetch_scenes(work), out)
    typer.echo(report.build_card(pipeline.Layout(work, out), repo, CODE_URL))
    if publish_:
        publish(out, repo, private)


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


Scratch = Annotated[Path, typer.Option(help="Node-local scratch directory.")]


@app.command("translate-plan")
def translate_plan(
    scratch: Scratch = Path("/tmp/georeset-plan"),
    workers: Workers = 8,
    unit: Annotated[list[str] | None, typer.Option(help="Plan only this unit (repeatable).")] = None,
) -> None:
    """Route every row of the release and upload the texts to translate to the bucket (CPU job)."""
    translation.plan_all(scratch, workers, unit)


@app.command("translate-pending")
def translate_pending() -> None:
    """Work items for g5k-dispatch: the work shards left in the bucket."""
    typer.echo(translation.pending(Bucket()))


@app.command("translate-work")
def translate_work(
    scratch: Scratch = Path("/tmp/georeset-translate"),
    offset: Annotated[int, typer.Option(help="Where in the shard list this job starts.")] = 0,
    minutes: Annotated[float, typer.Option(help="Time budget: no shard starts that may not end in it.")] = 50,
) -> None:
    """Translate work shards on this node's GPU until the time budget runs out (GPU job)."""
    typer.echo(f"{translation.translate(scratch, offset, minutes * 60)} shards translated")


@app.command("translate-assemble")
def translate_assemble(
    work: Work = Path("/tmp/georeset-work"),
    out: Out = Path("/tmp/georeset-out"),
    scratch: Scratch = Path("/tmp/georeset-assemble"),
) -> None:
    """Copy the release, add the translation columns, write the card, open a pull request (CPU job)."""
    translation.assemble_all(out, work, scratch)
    typer.echo(report.build_card(pipeline.Layout(work, out), REPO_ID, CODE_URL))
    typer.echo(translation.propose(out))


@app.command("translate-upload-model")
def translate_upload_model(folder: Path) -> None:
    """Upload NLLB converted for CTranslate2 to the bucket (``scripts/g5k/convert_model.sh``)."""
    translation.upload_model(folder, Bucket())


@app.command("translate-finalize")
def translate_finalize(pr: Annotated[int, typer.Option(help="Number of the reviewed pull request.")]) -> None:
    """Merge the translations' pull request, tag v1.1, delete the bucket."""
    translation.finalize(pr, Bucket())


@app.command("extras-propose")
def extras_propose(
    work: Work = Path("/tmp/georeset-work"),
    out: Out = Path("/tmp/georeset-out"),
) -> None:
    """Copy the fused dataset's embeddings and EUNIS label shares into the release, check them, write the
    card, open a pull request (CPU job)."""
    extras.assemble(out, work)
    typer.echo(report.build_card(pipeline.Layout(work, out), REPO_ID, CODE_URL))
    typer.echo(extras.propose(out))


@app.command("extras-finalize")
def extras_finalize(pr: Annotated[int, typer.Option(help="Number of the reviewed pull request.")]) -> None:
    """Merge the embeddings and EUNIS labels' pull request, tag v1.2."""
    extras.finalize(pr)


EMBED_BUCKET = embedding.BUCKET


@app.command("embed-plan")
def embed_plan(scratch: Scratch = Path("/tmp/georeset-embed-plan")) -> None:
    """Upload the English texts to embed to the bucket as work shards (CPU job)."""
    typer.echo(f"{embedding.plan(scratch, Bucket(EMBED_BUCKET))} texts to embed")


@app.command("embed-pending")
def embed_pending() -> None:
    """Work items for g5k-dispatch: the work shards left in the bucket."""
    typer.echo(embedding.pending(Bucket(EMBED_BUCKET)))


@app.command("embed-work")
def embed_work(
    scratch: Scratch = Path("/tmp/georeset-embed"),
    offset: Annotated[int, typer.Option(help="Where in the shard list this job starts.")] = 0,
    minutes: Annotated[float, typer.Option(help="Time budget: no shard starts that may not end in it.")] = 50,
) -> None:
    """Embed work shards on this node's GPU until the time budget runs out (GPU job)."""
    typer.echo(f"{embedding.work(scratch, offset, minutes * 60)} shards embedded")


@app.command("embed-assemble")
def embed_assemble(
    work: Work = Path("/tmp/georeset-work"),
    out: Out = Path("/tmp/georeset-out"),
) -> None:
    """Add the vectors to the release as embeddings_en, check them, write the card, open a pull request
    (CPU job)."""
    embedding.assemble(out, work, Bucket(EMBED_BUCKET))
    typer.echo(report.build_card(pipeline.Layout(work, out), REPO_ID, CODE_URL))
    typer.echo(embedding.propose(out))


@app.command("embed-finalize")
def embed_finalize(pr: Annotated[int, typer.Option(help="Number of the reviewed pull request.")]) -> None:
    """Merge the English vectors' pull request, tag v1.3, delete the bucket."""
    embedding.finalize(pr, Bucket(EMBED_BUCKET))
