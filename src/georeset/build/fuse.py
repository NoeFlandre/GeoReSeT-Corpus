"""Fuse a published corpus release with the geo-rag scenes into one dataset folder."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from georeset import hub
from georeset.build.write import write_shards, write_stream
from georeset.sources import scenes

CORPUS_FOLDERS = ("data", "unlabelled", "polygons")
SCENES_PER_SHARD = 100_000
SCENE_PHOTOS_PER_SHARD = 1_000_000
JOIN_CHUNK_ROWS = 500_000  # photos given their scene text at a time, to bound memory
SCENE_PHOTO_COLUMNS = (
    "scene_id", "text", "lulc_label", "lat", "lon", "h3_cell", "country", "continent", "captured_at",
    "season", "time_of_day", "koppen", "platform", "photo_id", "url", "photo_license",
)  # fmt: skip
_PINNED = re.compile(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)/tree/([0-9a-f]{40})")


def card_revisions(card: str) -> dict[str, str]:
    """The upstream ``repo -> commit`` pins listed in a published card."""
    return dict(_PINNED.findall(card))


def fetch_corpus(repo: str, rev: str, out: Path, work: Path) -> None:
    """Copy the corpus files of ``repo`` at ``rev`` into ``out``; keep its upstream pins for the card."""
    hub.snapshot(repo, rev, out, [f"{folder}/*" for folder in CORPUS_FOLDERS])
    card = hub.download(repo, "README.md", rev, work / "corpus").read_text()
    work.mkdir(parents=True, exist_ok=True)  # pragma: no mutate (idempotent setup)
    (work / "revisions.json").write_text(json.dumps(card_revisions(card), indent=2))


def fetch_scenes(work: Path) -> Path:
    """Download the pinned raw geo-rag files; record the commit for the card."""
    rev = hub.revision(scenes.RAW_REPO)
    folder = work / "scenes"
    for name in (scenes.PHOTOS_FILE, scenes.CLUSTERS_FILE):
        hub.download(scenes.RAW_REPO, name, rev, folder)
    (work / "scenes.json").write_text(json.dumps({scenes.RAW_REPO: rev}, indent=2))
    return folder


def _with_text(photos: pl.DataFrame, texts: pl.DataFrame) -> Iterator[pl.DataFrame]:
    """``photos`` (sorted by scene) chunk by chunk, each row given its scene's text and label."""
    for chunk in photos.iter_slices(JOIN_CHUNK_ROWS):
        joined = chunk.join(texts, on="scene_id", maintain_order="left")  # pragma: no mutate (order)
        yield joined.select(SCENE_PHOTO_COLUMNS)


def build_scenes(raw: Path, out: Path) -> None:
    """``scenes``: one row per description; ``scene_photos``: one row per photo, text repeated."""
    clusters = pl.scan_parquet(raw / scenes.CLUSTERS_FILE)
    scene_rows = scenes.scenes(clusters).collect(engine="streaming")
    described = scene_rows.lazy().select("scene_id")
    all_photos = scenes.photos(pl.scan_parquet(raw / scenes.PHOTOS_FILE), clusters)
    kept = all_photos.join(described, on="scene_id", how="semi")  # pragma: no mutate (= inner join)
    photos = kept.sort("scene_id", "platform", "photo_id").collect(engine="streaming")
    table = scenes.with_spread(scene_rows.lazy(), photos.lazy()).collect()
    write_shards(table, out / "scenes", SCENES_PER_SHARD)
    texts = table.select("scene_id", "text", "lulc_label")
    write_stream(_with_text(photos, texts), photos.height, out / "scene_photos", SCENE_PHOTOS_PER_SHARD)
