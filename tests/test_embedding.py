import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset import embedding


def _rows():
    """a/b/c relevant nllb (b repeats a's translation), d original, e not relevant, f unsplit."""
    return pl.DataFrame(
        {
            "id": ["b", "a", "c", "d", "e", "f"],
            "text": ["x1", "x2", "y", "Hello", "z", "w"],
            "text_en": ["same", "same", "other", "Hello", "zz", None],
            "translation": ["nllb", "nllb", "nllb", "original", "nllb", "unsplit"],
            "landuse": [True, True, True, True, False, True],
            "lat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "lon": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )


def test_selection_keeps_relevant_translations_once_with_the_smallest_id():
    rows = embedding.selection(_rows().lazy()).collect()
    assert rows.rows() == [("a", "same", 2.0, 8.0), ("c", "other", 3.0, 9.0)]
    assert rows.columns == ["id", "text", "lat", "lon"]


def test_work_shards_split_rows_and_name_them_in_the_bucket(tmp_path):
    rows = pl.DataFrame({"id": [str(i) for i in range(5)]})
    pairs = embedding.work_shards(rows, tmp_path / "w", size=2)
    assert [remote for _, remote in pairs] == [
        "work/00000.parquet",
        "work/00001.parquet",
        "work/00002.parquet",
    ]
    assert [pl.read_parquet(local)["id"].to_list() for local, _ in pairs] == [["0", "1"], ["2", "3"], ["4"]]
    assert embedding.ROWS_PER_SHARD == 50_000


def _release_data(folder):
    path = folder / "release" / "data" / "wikipedia" / "fra_Latn"
    path.mkdir(parents=True)
    _rows().write_parquet(path / "train-0.parquet")


def test_plan_uploads_the_shards_then_the_summary_and_cleans_up(bucket, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    calls = []
    monkeypatch.setattr(embedding.hub, "revision_of", lambda repo, ref: f"{repo}@{ref}")
    monkeypatch.setattr(embedding.hub, "snapshot", lambda *a: calls.append(a) or _release_data(scratch))
    assert embedding.plan(scratch, bucket) == 2
    rev = f"{embedding.REPO}@v1.2"
    assert calls == [(embedding.REPO, rev, scratch / "release", ["data/*"])]
    assert bucket.calls == [("create",), ("put", ["work/00000.parquet"]), ("put", ["plan.json"])]
    assert json.loads(bucket.data["plan.json"]) == {"texts": 2, "release": rev}
    assert pl.read_parquet(io.BytesIO(bucket.data["work/00000.parquet"]))["text"].to_list() == [
        "same",
        "other",
    ]
    assert not scratch.exists()


def test_plan_done_once_reads_its_summary(bucket, tmp_path, monkeypatch):
    bucket.data["plan.json"] = json.dumps({"texts": 9, "release": "r"}).encode()
    monkeypatch.setattr(embedding.hub, "snapshot", lambda *a: pytest.fail("planned again"))
    assert embedding.plan(tmp_path, bucket) == 9


def test_pending_counts_the_work_left_and_done(bucket):
    bucket.data.update({"work/00001.parquet": b"", "results/00000.parquet": b"", "plan.json": b""})
    assert json.loads(embedding.pending(bucket)) == [
        {"key": "embed", "memory_gb": 16, "remaining": 1, "fp16_ok": True, "done": 1}
    ]


@pytest.mark.parametrize(
    ("given", "used"), [("float16", "float32"), ("bfloat16", "bfloat16"), (None, "bfloat16")]
)
def test_torch_dtype_avoids_float16(given, used):
    assert embedding.torch_dtype(given) == used


def test_embed_shard_stores_float16_vectors_like_embeddings():
    work = pl.DataFrame({"id": ["a", "c"], "text": ["same", "other"], "lat": [2.0, 3.0], "lon": [8.0, 9.0]})
    seen = []

    def encode(texts):
        seen.append(texts)
        return np.array([[0.5, 0.25, 1.0], [1.0, 0.0, 0.125]], dtype=np.float32)

    table = embedding.embed_shard(encode, work)
    assert seen == [["same", "other"]]
    assert table.column_names == ["id", "lat", "lon", "vector"]
    assert table.schema.field("vector").type == pa.list_(pa.float16(), 3)
    assert table.column("vector").to_pylist() == [[0.5, 0.25, 1.0], [1.0, 0.0, 0.125]]
    assert table.column("id").to_pylist() == ["a", "c"]


def test_result_names_number_the_results_in_order():
    assert embedding.result_names(["results/00001.parquet", "results/00000.parquet"]) == [
        ("results/00000.parquet", "train-00000-of-00002.parquet"),
        ("results/00001.parquet", "train-00001-of-00002.parquet"),
    ]


def test_fetch_needs_every_shard_embedded(bucket, tmp_path, monkeypatch):
    bucket.data["work/00000.parquet"] = b""
    monkeypatch.setattr(embedding.fuse, "fetch_corpus", lambda *a: pytest.fail("fetched"))
    with pytest.raises(RuntimeError, match=r"^work shards are left: embedding is not finished$"):
        embedding.fetch(tmp_path / "out", tmp_path / "work", bucket)


def test_fetch_gets_the_release_its_context_and_the_vectors(bucket, tmp_path, monkeypatch):
    out, work = tmp_path / "out", tmp_path / "work"
    bucket.data.update({"results/00001.parquet": b"1", "results/00000.parquet": b"0"})
    calls = []
    monkeypatch.setattr(embedding.hub, "revision_of", lambda repo, ref: f"{repo}@{ref}")
    monkeypatch.setattr(embedding.fuse, "fetch_corpus", lambda *a: calls.append(("corpus", *a)))
    monkeypatch.setattr(embedding.hub, "snapshot", lambda *a: calls.append(("snapshot", *a)))
    monkeypatch.setattr(embedding.extras, "restore_context", lambda w: calls.append(("context", w)))
    embedding.fetch(out, work, bucket)
    rev = f"{embedding.REPO}@v1.2"
    assert calls == [
        ("corpus", embedding.REPO, rev, out, work),
        ("snapshot", embedding.REPO, rev, out, ["embeddings/*", "polygons_labels_shares/*"]),
        ("context", work),
    ]
    folder = out / "embeddings_en"
    assert sorted(p.name for p in folder.iterdir()) == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    assert (folder / "train-00001-of-00002.parquet").read_bytes() == b"1"


def _vectors(rows):
    ids, lats, lons = zip(*rows, strict=True)
    return pl.DataFrame({"id": list(ids), "lat": list(lats), "lon": list(lons)})


def _release(out, vectors):
    data = out / "data" / "wikipedia" / "fra_Latn"
    data.mkdir(parents=True)
    _rows().write_parquet(data / "train-0.parquet")
    (out / "embeddings_en").mkdir()
    _vectors(vectors).write_parquet(out / "embeddings_en" / "train-0.parquet")
    return out


FIT = [("a", 2.0, 8.0), ("c", 3.0, 9.0)]


def test_check_passes_when_the_vectors_fit(tmp_path):
    assert embedding.check(_release(tmp_path, FIT)) == []


@pytest.mark.parametrize(
    ("vectors", "problems"),
    [
        ([*FIT, ("a", 2.0, 8.0)], ["vectors sharing an id: 1", "vectors sharing a translation: 1"]),
        ([*FIT, ("d", 4.0, 10.0)], ["vectors of no relevant translated sentence: 1"]),
        ([*FIT, ("e", 5.0, 11.0)], ["vectors of no relevant translated sentence: 1"]),
        ([("a", 2.0, 8.0), ("c", 3.0, 0.0)], ["vectors away from their sentence: 1"]),
        ([("a", 2.0, 8.0), ("c", 0.0, 9.0)], ["vectors away from their sentence: 1"]),
        ([("a", 2.0, 8.0)], ["translations without a vector: 1"]),
    ],
    ids=["dup-id", "original", "irrelevant", "moved-lon", "moved-lat", "missing"],
)
def test_check_reports_what_does_not_fit(tmp_path, vectors, problems):
    assert embedding.check(_release(tmp_path, vectors)) == problems


def test_check_reports_two_vectors_of_one_translation(tmp_path):
    vectors = [("a", 2.0, 8.0), ("b", 1.0, 7.0), ("c", 3.0, 9.0)]
    assert embedding.check(_release(tmp_path, vectors)) == ["vectors sharing a translation: 1"]


def test_check_counts_with_thousands_separators(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embedding, "_checks", lambda out: {"x": pl.LazyFrame({"n": [1234]}), "y": pl.LazyFrame({"n": [0]})}
    )
    assert embedding.check(tmp_path) == ["x: 1,234"]


def test_check_scans_each_folder_by_its_exact_name(tmp_path, monkeypatch):
    out = _release(tmp_path, FIT)
    scanned, scan = [], pl.scan_parquet
    monkeypatch.setattr(embedding.pl, "scan_parquet", lambda path: scanned.append(path) or scan(path))
    embedding.check(out)
    assert {str(p.relative_to(out)) for p in scanned} == {"data/*/*/*.parquet", "embeddings_en/*.parquet"}


def test_assemble_refuses_vectors_that_do_not_fit(tmp_path, monkeypatch):
    fetched, checked = [], []
    monkeypatch.setattr(embedding, "fetch", lambda out, work, bucket: fetched.append((out, work, bucket)))
    monkeypatch.setattr(embedding, "check", lambda out: checked.append(out) or ["a: 1", "b: 2"])
    with pytest.raises(ValueError, match=r"^a: 1; b: 2$"):
        embedding.assemble(tmp_path / "out", tmp_path / "work", "bucket")
    assert fetched == [(tmp_path / "out", tmp_path / "work", "bucket")]
    assert checked == [tmp_path / "out"]
    monkeypatch.setattr(embedding, "check", lambda out: [])
    embedding.assemble(tmp_path / "out", tmp_path / "work", "bucket")


def test_propose_leaves_the_release_files_out(tmp_path, monkeypatch):
    for folder in ("data/wikipedia", "unlabelled", "embeddings", "polygons_labels_shares", "embeddings_en"):
        (tmp_path / folder).mkdir(parents=True)  # no polygons nor assets: a missing folder is skipped
    seen = []
    monkeypatch.setattr(
        embedding.hub,
        "propose",
        lambda out, repo, title: seen.append((sorted(p.name for p in out.iterdir()), repo, title)) or "url",
    )
    assert embedding.propose(tmp_path) == "url"
    assert seen == [(["embeddings_en"], embedding.REPO, embedding.PR_TITLE)]


def test_finalize_merges_tags_v1_3_then_deletes_the_bucket(bucket, monkeypatch):
    calls = []
    monkeypatch.setattr(embedding.hub, "merge", lambda repo, n: calls.append(("merge", repo, n)))
    monkeypatch.setattr(embedding.hub, "tag", lambda repo, name: calls.append(("tag", repo, name)))
    embedding.finalize(7, bucket)
    assert calls == [("merge", embedding.REPO, 7), ("tag", embedding.REPO, "v1.3")]
    assert bucket.calls == [("remove",)]


def test_the_vectors_read_back_as_themes_reads_embeddings(tmp_path):
    from georeset.publish import themes

    work = pl.DataFrame({"id": ["a"], "text": ["t"], "lat": [1.0], "lon": [2.0]})
    table = embedding.embed_shard(lambda texts: np.ones((1, 4), dtype=np.float32), work)
    pq.write_table(table, tmp_path / "v.parquet")
    ids, vectors = themes.read_vectors(tmp_path / "v.parquet")
    assert ids.to_list() == ["a"]
    assert vectors.tolist() == [[1.0] * 4]
    assert Path(tmp_path / "v.parquet").exists()
