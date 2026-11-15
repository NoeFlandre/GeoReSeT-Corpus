import json

import polars as pl

from georeset.translate import jobs, route


def test_release_resolves_the_tag(monkeypatch):
    monkeypatch.setattr(jobs.hub, "revision_of", lambda repo, ref: f"{repo}@{ref}")
    assert jobs.release() == "NoeFlandre/GeoReSeT-Corpus@v1.0"


def _shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=["id", "text", "language", "source"], orient="row")
    frame.write_parquet(path)


def _identify(texts):
    return ["fra_Latn"] * len(texts), [0.1] * len(texts)


def test_plan_unit_uploads_the_summary_last_and_cleans_up(bucket, tmp_path, monkeypatch):
    def snapshot(repo, rev, dest, patterns):
        assert (repo, rev, patterns) == (jobs.REPO, "sha", ["data/*/fra_Latn/*.parquet"])
        _shard(dest / "data/wikipedia/fra_Latn/train-0.parquet", [("1", "Salut", "fra_Latn", "wikipedia")])
        _shard(dest / "data/website/fra_Latn/train-0.parquet", [("2", "Oui", "fra_Latn", "website")])

    monkeypatch.setattr(jobs.hub, "snapshot", snapshot)
    jobs.plan_unit("fra_Latn", "sha", bucket, tmp_path, _identify)
    assert bucket.calls == [
        ("put", ["work/fra_Latn/00000.parquet", "routes/fra_Latn.parquet"]),
        ("put", ["plan/fra_Latn.json"]),
    ]
    assert json.loads(bucket.data["plan/fra_Latn.json"])["distinct"] == 2
    assert list(tmp_path.iterdir()) == []
    jobs.plan_unit("fra_Latn", "sha", bucket, tmp_path, _identify)
    assert len(bucket.calls) == 2


def test_pending_counts_the_work_shards(bucket):
    bucket.data.update({"work/a/0.parquet": b"", "work/a/1.parquet": b"", "results/a/2.parquet": b""})
    assert json.loads(jobs.pending(bucket)) == [
        {"key": "nllb", "memory_gb": jobs.MEMORY_GB, "remaining": 2, "fp16_ok": True}
    ]


def test_fetch_and_upload_model(bucket, tmp_path):
    source = tmp_path / "converted"
    (source / "sub").mkdir(parents=True)
    (source / "model.bin").write_bytes(b"w")
    (source / "config.json").write_bytes(b"c")
    jobs.upload_model(source, bucket)
    assert bucket.calls[0] == ("create",)
    assert sorted(bucket.data) == ["model/config.json", "model/model.bin"]
    folder = jobs.fetch_model(bucket, tmp_path / "scratch")
    assert folder == tmp_path / "scratch" / "model"
    assert (folder / "model.bin").read_bytes() == b"w"


def test_assemble_unit_rewrites_its_shards_with_the_results(bucket, tmp_path):
    out = tmp_path / "out"
    _shard(out / "data/wikipedia/fra_Latn/train-0.parquet", [("1", "Salut", "fra_Latn", "wikipedia")])
    _shard(out / "data/wikipedia/deu_Latn/train-0.parquet", [("2", "Hallo", "deu_Latn", "wikipedia")])
    result = pl.DataFrame(
        {"key": [route.key("fra_Latn", "Salut")], "text_en": ["Hi"], "translation_flags": [[]]},
        schema={"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)},
    )
    result.write_parquet(tmp_path / "r.parquet")
    pl.DataFrame(schema={"id": pl.String, "detected": pl.String}).write_parquet(tmp_path / "o.parquet")
    bucket.data["results/fra_Latn/00000.parquet"] = (tmp_path / "r.parquet").read_bytes()
    bucket.data["routes/fra_Latn.parquet"] = (tmp_path / "o.parquet").read_bytes()
    assert jobs.assemble_unit("fra_Latn", out, bucket, tmp_path / "scratch") == 1
    assert pl.read_parquet(out / "data/wikipedia/fra_Latn/train-0.parquet")["text_en"].to_list() == ["Hi"]
    assert "text_en" not in pl.read_parquet(out / "data/wikipedia/deu_Latn/train-0.parquet").columns
    assert list((tmp_path / "scratch").iterdir()) == []


def test_summary_adds_up_the_plans(bucket, tmp_path):
    bucket.data["plan/a.json"] = b'{"distinct": 2}'
    bucket.data["plan/b.json"] = b'{"distinct": 3}'
    assert jobs.summary(bucket, tmp_path, "sha") == {
        "distinct": 5,
        "model": "facebook/nllb-200-3.3B",
        "revision": "1a07f7d195896b2114afcb79b7b57ab512e7b43e",
        "threshold": route.THRESHOLD,
        "release": "sha",
    }


def test_planned_units(bucket):
    bucket.data.update({"plan/fra_Latn.json": b"", "plan/unlabelled.json": b""})
    assert jobs.planned(bucket) == {"fra_Latn", "unlabelled"}
