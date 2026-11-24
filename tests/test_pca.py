from pathlib import Path

import polars as pl

from georeset import pca
from georeset.embedding import ROWS_PER_SHARD
from georeset.publish import themes


def _shards(folder: Path, names: list[str]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return sorted(folder / name for name in names)


def test_pools_pair_each_sentence_shard_with_its_vectors_and_language():
    landuse = [Path("s/a.parquet"), Path("s/b.parquet")]
    translated = [Path("t/0.parquet")]
    vectors = Path("v")
    _shards(vectors / "embeddings", ["train-a.parquet", "train-b.parquet"])
    _shards(vectors / "embeddings_en", ["train-0.parquet"])
    (landuse_pool, translated_pool) = pca.pools(landuse, translated, vectors)
    assert landuse_pool[:2] == ("landuse", landuse)
    assert landuse_pool[3] == ["", ""]
    name, sentences, vector_files, languages = translated_pool
    assert name == "english_pool"
    assert sentences == [*landuse, *translated]
    assert [p.name for p in vector_files] == ["train-a.parquet", "train-b.parquet", "train-0.parquet"]
    assert languages == ["eng_Latn", "eng_Latn", ""]


def test_write_translated_slices_the_rows_into_embeddings_en_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(pca.embedding, "ROWS_PER_SHARD", 2)
    rows = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": ["x", "y", "z"],
            "language": ["fra_Latn"] * 3,
            "lat": [1.0] * 3,
            "lon": [2.0] * 3,
        }
    )
    monkeypatch.setattr(pca.embedding, "selection", lambda data: rows.lazy())
    assert pca.write_translated(rows.lazy(), tmp_path / "t") == 2
    names = sorted(p.name for p in (tmp_path / "t").iterdir())
    assert names == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert pl.read_parquet(tmp_path / "t" / names[0])["id"].to_list() == ["a", "b"]
    assert pl.read_parquet(tmp_path / "t" / names[1])["id"].to_list() == ["c"]


def test_shard_size_is_the_embedding_one():
    assert ROWS_PER_SHARD == 50_000


def test_finalize_merges_tags_v1_4_then_deletes_the_bucket(monkeypatch):
    calls = []
    monkeypatch.setattr(pca.hub, "release", lambda repo, n, name: calls.append(("release", repo, n, name)))

    class Bucket:
        def remove(self):
            calls.append(("remove",))

    pca.finalize(9, Bucket())
    assert calls == [("release", pca.REPO, 9, "v1.4"), ("remove",)]


def test_fetch_themes_brings_each_corpus_outputs_back(tmp_path, monkeypatch):
    calls = []

    class Bucket:
        def files(self, prefix):
            return [f"{prefix}summary.json"]

        def get(self, files):
            calls.append(files)

    pca.fetch_themes(tmp_path, Bucket())
    assert calls == [
        [(f"themes/{corpus.name}/summary.json", tmp_path / "themes" / corpus.name / "summary.json")]
        for corpus in pca.CORPORA
    ]


def test_remove_deletes_folders_and_files(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"")
    (tmp_path / "f.txt").write_bytes(b"")
    pca._remove(tmp_path / "data")
    pca._remove(tmp_path / "f.txt")
    assert list(tmp_path.iterdir()) == []


def test_corpora_are_the_landuse_and_the_english_pool():
    assert pca.CORPORA == (themes.LANDUSE, themes.ENGLISH_POOL)
    assert themes.ENGLISH_POOL.only_language == ""
