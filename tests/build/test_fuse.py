import json
from pathlib import Path

import polars as pl
from conftest import SCENE_TEXT

from georeset.build import fuse
from georeset.sources import scenes

SHA_A, SHA_B = "a" * 40, "0123456789" * 4


def test_card_revisions():
    card = (
        f"* [`me/a` @ `aaaaaaa`](https://huggingface.co/datasets/me/a/tree/{SHA_A})\n"
        f"* [`me/b-c.d` @ `0123456`](https://huggingface.co/datasets/me/b-c.d/tree/{SHA_B})\n"
        "* [`me/x`](https://huggingface.co/datasets/me/x/tree/main)\n"
    )
    assert fuse.card_revisions(card) == {"me/a": SHA_A, "me/b-c.d": SHA_B}


def test_fetch_corpus(tmp_path, monkeypatch):
    calls = []
    readme = tmp_path / "README.md"
    readme.write_text(f"see https://huggingface.co/datasets/me/up/tree/{SHA_A}")
    monkeypatch.setattr(fuse.hub, "snapshot", lambda *a: calls.append(("snapshot", *a)))
    monkeypatch.setattr(fuse.hub, "download", lambda *a: calls.append(("download", *a)) or readme)
    work, out = tmp_path / "work", tmp_path / "out"
    fuse.fetch_corpus("me/corpus", "v1.0", out, work)
    assert calls == [
        ("snapshot", "me/corpus", "v1.0", out, ["data/*", "unlabelled/*", "polygons/*"]),
        ("download", "me/corpus", "README.md", "v1.0", work / "corpus"),
    ]
    assert (work / "revisions.json").read_text() == json.dumps({"me/up": SHA_A}, indent=2)


def test_fetch_scenes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fuse.hub, "revision", lambda repo: {scenes.RAW_REPO: SHA_B}[repo])
    monkeypatch.setattr(fuse.hub, "download", lambda *a: calls.append(a))
    tmp_path.joinpath("w").mkdir()
    folder = fuse.fetch_scenes(tmp_path / "w")
    assert folder == tmp_path / "w" / "scenes"
    assert calls == [
        (scenes.RAW_REPO, scenes.PHOTOS_FILE, SHA_B, folder),
        (scenes.RAW_REPO, scenes.CLUSTERS_FILE, SHA_B, folder),
    ]
    assert (tmp_path / "w" / "scenes.json").read_text() == json.dumps({scenes.RAW_REPO: SHA_B}, indent=2)


def _read(folder: Path) -> pl.DataFrame:
    return pl.read_parquet(folder / "*.parquet")


def test_build_scenes(raw_scenes, tmp_path, monkeypatch):
    monkeypatch.setattr(fuse, "JOIN_CHUNK_ROWS", 2)
    monkeypatch.setattr(fuse, "SCENE_PHOTOS_PER_SHARD", 2)
    out = tmp_path / "out"
    fuse.build_scenes(raw_scenes, out)
    table = _read(out / "scenes")
    assert table["scene_id"].to_list() == [3, 7]
    assert table["n_photos"].to_list() == [1, 2]
    assert table["countries"].to_list() == [["Mexico"], ["France"]]
    shards = sorted(p.name for p in (out / "scene_photos").iterdir())
    assert shards == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    photos = _read(out / "scene_photos")
    assert photos.columns == list(fuse.SCENE_PHOTO_COLUMNS)
    assert photos.select("scene_id", "platform", "photo_id", "text").rows() == [
        (3, "wikimedia", "3", "Dry bare earth."),
        (7, "flickr", "2", SCENE_TEXT),
        (7, "mapillary", "1", SCENE_TEXT),
    ]
    assert photos["lulc_label"].to_list() == ["Barren soil / Badlands", *["Transportation network"] * 2]


def test_with_text_keeps_photo_order_across_chunks(monkeypatch):
    monkeypatch.setattr(fuse, "JOIN_CHUNK_ROWS", 1)
    photos = pl.DataFrame({"scene_id": [2, 1], **{c: [c, c] for c in fuse.SCENE_PHOTO_COLUMNS[3:]}})
    texts = pl.DataFrame({"scene_id": [1, 2], "text": ["one", "two"], "lulc_label": ["L1", "L2"]})
    chunks = list(fuse._with_text(photos, texts))
    assert [c.height for c in chunks] == [1, 1]
    assert pl.concat(chunks).select("scene_id", "text", "lulc_label").rows() == [
        (2, "two", "L2"),
        (1, "one", "L1"),
    ]


def test_build_scenes_sorts_photos_by_id_within_platform(raw_scenes, tmp_path):
    for name, extra in [
        (scenes.PHOTOS_FILE, {"Photo_ID": "10", "Platform": "flickr", "photo_key": "flickr_10"}),
        (scenes.CLUSTERS_FILE, {"photo_key": "flickr_10"}),
    ]:
        table = pl.read_parquet(raw_scenes / name)
        row = table.head(2).tail(1).with_columns(**{k: pl.lit(v) for k, v in extra.items()})
        pl.concat([table, row]).write_parquet(raw_scenes / name)
    fuse.build_scenes(raw_scenes, tmp_path / "out")
    photos = _read(tmp_path / "out" / "scene_photos").filter(pl.col("scene_id") == 7)
    assert photos.select("platform", "photo_id").rows() == [
        ("flickr", "10"),
        ("flickr", "2"),
        ("mapillary", "1"),
    ]
