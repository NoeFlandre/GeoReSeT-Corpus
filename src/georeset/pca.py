"""PCA maps of the relevant sentences on the corpus: multilingual, and English (original and translated).

compute   (CPU job) the relevant sentences' shards of release v1.3, as ``themes`` reads them, with
          the vectors of ``embeddings`` (multilingual) and of ``embeddings`` plus ``embeddings_en``
          (English); ``themes.compute`` for both; the outputs go to the bucket ``BUCKET``
propose   (CPU job) the release's card with the two maps, its figures, and a pull request on the dataset
          with only the figures and the card; ``finalize`` merges it, tags v1.7 and deletes the bucket
"""

from pathlib import Path

import polars as pl

from georeset import embedding, hub
from georeset.build import extras
from georeset.build.pipeline import Layout
from georeset.publish import report, themes
from georeset.translate.store import Bucket

BUCKET = "NoeFlandre/georeset-pca"
REPO = extras.REPO
CODE_URL = "https://github.com/NoeFlandre/GeoReSeT-Corpus"
RELEASE, NEXT_RELEASE = "v1.3", "v1.7"
ENGLISH_LANGUAGE = "eng_Latn"
PR_TITLE = "Principal components section: maps and method in plain terms"
CORPORA = (themes.LANDUSE, themes.ENGLISH_POOL)  # the two maps, in the order of the card
KEPT_ON_PR = ("assets", "README.md")  # the others are the release's, unchanged


def write_translated(data: pl.LazyFrame, out: Path) -> None:
    """The English translations, in the shards and order of ``embeddings_en``."""
    rows = embedding.selection(data).collect(engine="streaming")
    shards = -(-rows.height // embedding.ROWS_PER_SHARD)
    out.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, rows.height, embedding.ROWS_PER_SHARD)):
        rows.slice(start, embedding.ROWS_PER_SHARD).write_parquet(out / themes.shard_name(i, shards))


def sentence_shards(
    scratch: Path,
) -> tuple[list[Path], list[Path]]:  # pragma: no cover (downloads the release)
    """Release v1.3's relevant sentences: the multilingual shards, then the translated ones."""
    rev = hub.revision_of(REPO, RELEASE)
    hub.snapshot(REPO, rev, scratch / "release", ["data/*"])
    data = pl.scan_parquet(scratch / "release" / "data" / "*" / "*" / "*.parquet")
    themes.write_sentences(data, scratch / "sentences", themes.LANDUSE)
    write_translated(data, scratch / "translated")
    return sorted((scratch / "sentences").glob("train-*.parquet")), sorted(
        (scratch / "translated").glob("train-*.parquet")
    )


def pools(
    landuse: list[Path], translated: list[Path], vectors: Path
) -> list[tuple[str, list[Path], list[Path], list[str]]]:
    """Each map's inputs: its corpus, sentence shards, vector shards and language filter per shard."""
    relevant = sorted((vectors / "embeddings").glob("train-*.parquet"))
    english = sorted((vectors / "embeddings_en").glob("train-*.parquet"))
    return [
        (themes.LANDUSE.name, landuse, relevant, [""] * len(landuse)),
        (
            themes.ENGLISH_POOL.name,
            [*landuse, *translated],
            [*relevant, *english],
            [ENGLISH_LANGUAGE] * len(landuse) + [""] * len(translated),
        ),
    ]


def compute(scratch: Path, work: Path, bucket: Bucket) -> None:  # pragma: no cover (Grid'5000 CPU job)
    bucket.create()
    landuse, translated = sentence_shards(scratch)
    rev = hub.revision_of(REPO, RELEASE)
    hub.snapshot(REPO, rev, scratch / "vectors", ["embeddings/*", "embeddings_en/*"])
    for name, sentences, vectors, languages in pools(landuse, translated, scratch / "vectors"):
        themes.compute(sentences, vectors, work, themes.CORPORA[name], languages=languages)
        out = work / "themes" / name
        files = sorted(out.glob("*"))
        bucket.put([(path, f"themes/{name}/{path.name}") for path in files])


def fetch_themes(work: Path, bucket: Bucket) -> None:
    """The outputs ``compute`` left in the bucket, back into ``work/themes``."""
    for corpus in CORPORA:
        files = bucket.files(f"themes/{corpus.name}/")
        bucket.get([(path, work / "themes" / corpus.name / Path(path).name) for path in files])


def propose(out: Path, work: Path, bucket: Bucket) -> str:  # pragma: no cover (Grid'5000 CPU job)
    """The card with the maps, and a pull request with only the figures and the card."""
    # The release, with the copied folders the card counts and the context its card keeps.
    extras.fetch_release(out, work, RELEASE, (*extras.EXTRAS_FOLDERS, "embeddings_en"))
    fetch_themes(work, bucket)
    report.build_card(Layout(work, out), REPO, CODE_URL)
    drop = [path.name for path in out.iterdir() if path.name not in KEPT_ON_PR]
    return hub.propose(out, REPO, PR_TITLE, drop)


def finalize(number: int, bucket: Bucket) -> None:
    """Merge the reviewed pull request, tag the release, delete the bucket."""
    hub.release(REPO, number, NEXT_RELEASE)
    bucket.remove()
