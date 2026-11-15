import json
from types import SimpleNamespace
from typing import ClassVar

import polars as pl
import pytest

from georeset.translate import assemble, jobs, plan, route


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
        {"key": "nllb", "memory_gb": jobs.MEMORY_GB, "remaining": 2, "fp16_ok": True, "done": 1}
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


def test_propose_leaves_polygons_out(tmp_path, monkeypatch):
    (tmp_path / "polygons").mkdir()
    (tmp_path / "data").mkdir()
    seen = []
    monkeypatch.setattr(
        jobs.hub, "propose", lambda out, repo, title: seen.append((out, repo, title)) or "url"
    )
    assert jobs.propose(tmp_path) == "url"
    assert seen == [(tmp_path, jobs.REPO, jobs.PR_TITLE)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]


def test_finalize_merges_tags_then_deletes_the_bucket(bucket, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.hub, "merge", lambda repo, n: calls.append(("merge", repo, n)))
    monkeypatch.setattr(jobs.hub, "tag", lambda repo, name: calls.append(("tag", repo, name)))
    jobs.finalize(4, bucket)
    assert calls == [("merge", jobs.REPO, 4), ("tag", jobs.REPO, "v1.1")]
    assert bucket.calls == [("remove",)]


def test_plan_unit_paths_and_report(bucket, tmp_path, monkeypatch):
    seen = {"dest": [], "rows": [], "write": []}

    def snapshot(repo, rev, dest, patterns):
        seen["dest"].append(dest)
        _shard(dest / "data/wikipedia/fra_Latn/train-0.parquet", [("1", "Salut", "fra_Latn", "wikipedia")])

    rows, write = plan.rows, plan.write
    monkeypatch.setattr(jobs.hub, "snapshot", snapshot)
    monkeypatch.setattr(plan, "rows", lambda path: seen["rows"].append(path) or rows(path))
    monkeypatch.setattr(plan, "write", lambda u, p, s: seen["write"].append(s) or write(u, p, s))
    printed = []
    monkeypatch.setattr(jobs, "print", lambda *a, **k: printed.append((a, k)), raising=False)
    jobs.plan_unit("fra_Latn", "sha", bucket, tmp_path, _identify)
    folder = tmp_path / "fra_Latn"
    assert seen == {
        "dest": [folder / "in"],
        "rows": [folder / "in/data/wikipedia/fra_Latn/train-0.parquet"],
        "write": [folder / "out"],
    }
    assert printed == [(("planned fra_Latn",), {"flush": True})]


def test_assemble_unit_reads_what_it_downloaded(bucket, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _shard(out / "data/wikipedia/fra_Latn/train-0.parquet", [("1", "Salut", "fra_Latn", "wikipedia")])
    result = pl.DataFrame(
        {"key": [route.key("fra_Latn", "Salut")], "text_en": ["Hi"], "translation_flags": [[]]},
        schema={"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)},
    )
    result.write_parquet(tmp_path / "r.parquet")
    pl.DataFrame(schema={"id": pl.String, "detected": pl.String}).write_parquet(tmp_path / "o.parquet")
    bucket.data["results/fra_Latn/00000.parquet"] = (tmp_path / "r.parquet").read_bytes()
    bucket.data["routes/fra_Latn.parquet"] = (tmp_path / "o.parquet").read_bytes()
    loaded, read, load, read_parquet = [], [], assemble.load_translations, pl.read_parquet
    monkeypatch.setattr(assemble, "load_translations", lambda paths: loaded.append(paths) or load(paths))
    monkeypatch.setattr(jobs.pl, "read_parquet", lambda path: read.append(path) or read_parquet(path))
    jobs.assemble_unit("fra_Latn", out, bucket, tmp_path / "scratch")
    folder = tmp_path / "scratch" / "fra_Latn"
    assert loaded == [[folder / "results/fra_Latn/00000.parquet"]]
    assert read == [folder / "results/fra_Latn/00000.parquet", folder / "routes/fra_Latn.parquet"]


def test_propose_removes_polygons_and_tolerates_their_absence(tmp_path, monkeypatch):
    removed = []
    monkeypatch.setattr(jobs.shutil, "rmtree", lambda path, **kwargs: removed.append((path, kwargs)))
    monkeypatch.setattr(jobs.hub, "propose", lambda out, repo, title: "url")
    jobs.propose(tmp_path)
    assert removed == [(tmp_path / "polygons", {"ignore_errors": True})]
    monkeypatch.undo()
    monkeypatch.setattr(jobs.hub, "propose", lambda out, repo, title: "url")
    assert jobs.propose(tmp_path) == "url"


class _Identify:
    def predict(self, texts):
        return _identify(texts)


def test_init_identify_loads_glotlid_once_per_process(monkeypatch):
    lid = _Identify()
    monkeypatch.setattr(jobs, "_IDENTIFY", None)
    monkeypatch.setattr(jobs.models, "GlotLid", lambda: lid)
    jobs._init_identify()
    assert lid.predict == jobs._IDENTIFY


def test_plan_in_process_needs_glotlid_and_plans_with_the_real_bucket(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "Bucket", lambda: "bucket")
    monkeypatch.setattr(jobs, "plan_unit", lambda *args: calls.append(args))
    monkeypatch.setattr(jobs, "_IDENTIFY", None)
    with pytest.raises(AssertionError):
        jobs._plan_in_process("fra_Latn", "sha", tmp_path)
    monkeypatch.setattr(jobs, "_IDENTIFY", _identify)
    jobs._plan_in_process("fra_Latn", "sha", tmp_path)
    assert calls == [("fra_Latn", "sha", "bucket", tmp_path, _identify)]


class _Pool:
    """Runs ``map`` in this process; records how it was made."""

    made: ClassVar[list] = []

    def __init__(self, workers, initializer):
        _Pool.made.append((workers, initializer))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *iterables):
        return map(fn, *iterables)


def test_plan_all_plans_the_units_not_planned_yet(bucket, tmp_path, monkeypatch):
    bucket.data["plan/deu_Latn.json"] = b"{}"
    done = []
    _Pool.made = []
    monkeypatch.setattr(jobs, "release", lambda: "sha")
    monkeypatch.setattr(jobs, "Bucket", lambda: bucket)
    monkeypatch.setattr(jobs, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(
        jobs.hub, "list_files", lambda repo, rev: ["data/w/deu_Latn/a.parquet", "data/w/fra_Latn/a.parquet"]
    )
    monkeypatch.setattr(
        jobs, "_plan_in_process", lambda unit, rev, scratch: done.append((unit, rev, scratch))
    )
    jobs.plan_all(tmp_path, 3)
    assert bucket.calls == [("create",)]
    assert _Pool.made == [(3, jobs._init_identify)]
    assert done == [("fra_Latn", "sha", tmp_path), ("unlabelled", "sha", tmp_path)]
    done.clear()
    jobs.plan_all(tmp_path, 1, ["deu_Latn", "ita_Latn"])
    assert done == [("ita_Latn", "sha", tmp_path)]


def test_translate_runs_a_worker_with_the_downloaded_model(tmp_path, monkeypatch):
    made = []

    class Worker:
        def __init__(self, *args):
            made.append(args)

        def run(self):
            return 7

    lid = SimpleNamespace(labels="labels")
    monkeypatch.setattr(jobs, "Bucket", lambda: "bucket")
    monkeypatch.setattr(jobs, "fetch_model", lambda bucket, scratch: (bucket, scratch, "model"))
    monkeypatch.setattr(jobs.models, "Nllb", lambda folder: ("nllb", folder))
    monkeypatch.setattr(jobs.models, "GlotLid", lambda: lid)
    monkeypatch.setattr(jobs.worker, "Worker", Worker)
    assert jobs.translate(tmp_path, 2, 60.0) == 7
    assert made == [("bucket", ("nllb", ("bucket", tmp_path, "model")), "labels", tmp_path, 2, 60.0)]


def test_assemble_all_needs_every_shard_translated(bucket, tmp_path, monkeypatch):
    bucket.data["work/a/0.parquet"] = b""
    monkeypatch.setattr(jobs, "release", lambda: "sha")
    monkeypatch.setattr(jobs, "Bucket", lambda: bucket)
    monkeypatch.setattr(jobs.fuse, "fetch_corpus", lambda *args: pytest.fail("fetched"))
    with pytest.raises(RuntimeError, match="work shards are left: translation is not finished"):
        jobs.assemble_all(tmp_path / "out", tmp_path / "work", tmp_path / "scratch")


def test_assemble_all_assembles_each_unit_then_writes_the_summary(bucket, tmp_path, monkeypatch):
    out, work, scratch = tmp_path / "out", tmp_path / "work", tmp_path / "scratch"
    work.mkdir()
    calls, printed = [], []
    monkeypatch.setattr(jobs, "release", lambda: "sha")
    monkeypatch.setattr(jobs, "Bucket", lambda: bucket)
    monkeypatch.setattr(jobs.fuse, "fetch_corpus", lambda *args: calls.append(("fetch", *args)))
    monkeypatch.setattr(
        jobs.hub,
        "list_files",
        lambda repo, rev: calls.append(("list", repo, rev)) or ["data/w/fra_Latn/a.parquet"],
    )
    monkeypatch.setattr(jobs, "assemble_unit", lambda *args: calls.append(("unit", *args)) or 2)
    monkeypatch.setattr(jobs, "summary", lambda *args: calls.append(("summary", *args)) or {"distinct": 1})
    monkeypatch.setattr(jobs, "print", lambda *a, **k: printed.append((a, k)), raising=False)
    jobs.assemble_all(out, work, scratch)
    assert calls == [
        ("fetch", jobs.REPO, "sha", out, work),
        ("list", jobs.REPO, "sha"),
        ("unit", "fra_Latn", out, bucket, scratch),
        ("unit", "unlabelled", out, bucket, scratch),
        ("summary", bucket, scratch, "sha"),
    ]
    assert printed == [
        (("fra_Latn: 2 shards",), {"flush": True}),
        (("unlabelled: 2 shards",), {"flush": True}),
    ]
    assert (work / "translation.json").read_text() == '{\n  "distinct": 1\n}'
