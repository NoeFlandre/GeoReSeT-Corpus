"""Add the fused dataset's sentence embeddings and polygon EUNIS label shares to the corpus' latest release.

The folders are copied as they are, checked against the release, and proposed in a pull request with the
new card; the release's own files are left out of it, unchanged.
"""

import json
import re
from pathlib import Path

import polars as pl

from georeset import hub
from georeset.build import fuse
from georeset.label import attach
from georeset.publish.card import EXTRAS_FOLDERS
from georeset.translate import models, route

REPO = "NoeFlandre/GeoReSeT-Corpus"
FUSED_REPO = "NoeFlandre/GeoReSeT-Corpus-Fused"
RELEASE, NEXT_RELEASE = "v1.1", "v1.2"
PR_TITLE = "Add the sentence embeddings and polygon EUNIS label shares of GeoReSeT-Corpus-Fused"
UNCHANGED = ("data", "unlabelled", "polygons", "assets")
_DISTINCT = re.compile(r"\(([\d,]+) distinct texts from")


def translation_summary(card: str) -> dict:
    """``work/translation.json`` of a translated release: count from its card, settings from the code."""
    (distinct,) = _DISTINCT.findall(card)
    return {
        "distinct": int(distinct.replace(",", "")),
        "model": models.NLLB_REPO,
        "revision": models.NLLB_REVISION,
        "threshold": route.THRESHOLD,
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2))


def restore_context(work: Path) -> None:
    """What the card needs that only the release's card keeps (``work/corpus/README.md``): the labels'
    upstream pins, the fused dataset's pin if any, and the translation summary.
    """
    card = (work / "corpus" / "README.md").read_text()
    pins = fuse.card_revisions(card)
    if FUSED_REPO in pins:
        _write(work / "fused.json", {FUSED_REPO: pins.pop(FUSED_REPO)})
    _write(work / "revisions.json", pins)
    _write(work / "translation.json", translation_summary(card))


def fetch_release(out: Path, work: Path, release: str, folders: tuple[str, ...] = ()) -> None:
    """The ``release`` tag into ``out`` (its corpus folders, then ``folders``); its context into ``work``."""
    rev = hub.revision_of(REPO, release)
    fuse.fetch_corpus(REPO, rev, out, work)
    if folders:
        hub.snapshot(REPO, rev, out, [f"{folder}/*" for folder in folders])
    restore_context(work)


def fetch(out: Path, work: Path) -> None:
    """The release and fused folders into ``out``; their pins and the translation summary into ``work``."""
    fetch_release(out, work, RELEASE)
    fused = hub.revision(FUSED_REPO)
    hub.snapshot(FUSED_REPO, fused, out, [f"{folder}/*" for folder in EXTRAS_FOLDERS])
    _write(work / "fused.json", {FUSED_REPO: fused})


def _scan(out: Path, folder: str) -> pl.LazyFrame:
    return pl.scan_parquet(out / folder / "*.parquet")


def _checks(out: Path) -> dict[str, pl.LazyFrame]:
    """Each check of the copied folders against the release, as a count of the rows that fail it."""
    relevant = attach.landuse(pl.scan_parquet(out / "data" / "*" / "*" / "*.parquet"), out).filter(
        pl.col("landuse")
    )
    vectors = _scan(out, "embeddings").select("id", "lat", "lon")
    placed = vectors.join(relevant.select("id", "text", "lat", "lon"), on="id", suffix="_text")
    shares = _scan(out, "polygons_labels_shares").select("polygon_id")
    polygons = _scan(out, "polygons").select("polygon_id")
    moved = (pl.col("lat") != pl.col("lat_text")) | (pl.col("lon") != pl.col("lon_text"))
    unembedded = relevant.select("text").join(placed.select("text"), on="text", how="anti")
    return {
        "embeddings sharing an id": vectors.select(pl.len() - pl.col("id").n_unique()),
        "embeddings of no relevant sentence": vectors.join(relevant, on="id", how="anti").select(pl.len()),
        "embeddings away from their sentence": placed.filter(moved).select(pl.len()),
        "relevant texts without an embedding": unembedded.select(pl.col("text").n_unique()),
        "label shares sharing a polygon": shares.select(pl.len() - pl.col("polygon_id").n_unique()),
        "label shares of no polygon": shares.join(polygons, on="polygon_id", how="anti").select(pl.len()),
        "polygons without label shares": polygons.join(shares, on="polygon_id", how="anti").select(pl.len()),
    }


def check(out: Path) -> list[str]:
    """What is wrong with the copied folders against the release in ``out``; empty when they fit it."""
    counts = {name: lazy.collect(engine="streaming").item() for name, lazy in _checks(out).items()}
    return [f"{name}: {n:,}" for name, n in counts.items() if n]


def assemble(out: Path, work: Path) -> None:
    """Fetch the release and the fused folders, and refuse them unless they fit."""
    fetch(out, work)
    if problems := check(out):
        raise ValueError("; ".join(problems))


def propose(out: Path) -> str:
    """Open a pull request with the copied folders and the new card; the release's own files are left out."""
    return hub.propose(out, REPO, PR_TITLE, UNCHANGED)


def finalize(number: int) -> None:
    """Merge the reviewed pull request and tag the release."""
    hub.release(REPO, number, NEXT_RELEASE)
