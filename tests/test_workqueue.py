import io

import polars as pl
import pytest

from georeset import workqueue


def _work(rows):
    return pl.DataFrame(
        rows, schema={"key": pl.Binary, "translation_language": pl.String, "text": pl.String}, orient="row"
    )


def _upper(work):
    """A transform: the shard's keys with their text upper-cased."""
    return work.select("key", pl.col("text").str.to_uppercase())


def test_claims_keep_recent_ones():
    paths = ["claims/fra_Latn/00001.1000", "claims/fra_Latn/00002.100", "claims/a.b/00003.999"]
    assert workqueue.claims(paths, now=1000 + workqueue.CLAIM_S - 1) == {"fra_Latn/00001"}
    assert workqueue.claims(paths, now=1000.0) == {"fra_Latn/00001", "fra_Latn/00002", "a.b/00003"}


def test_choose_skips_taken_shards_and_spreads_by_offset():
    work = ["work/a/00000.parquet", "work/a/00001.parquet", "work/b/00000.parquet"]
    assert workqueue.choose(work, {"a/00001"}, 0) == "a/00000"
    assert workqueue.choose(work, {"a/00001"}, 3) == "b/00000"
    assert workqueue.choose(work, {"a/00000", "a/00001", "b/00000"}, 0) is None


def test_may_start_only_if_the_longest_shard_still_fits():
    assert workqueue.may_start(10, 20, 31)
    assert not workqueue.may_start(10, 20, 30)


class Clock:
    def __init__(self, step):
        self.now, self.step = 1000.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _bucket_with(bucket, shards):
    for name, rows in shards.items():
        path = f"work/{name}.parquet"
        frame = _work(rows)

        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        bucket.data[path] = buffer.getvalue()


def test_worker_translates_free_shards_then_frees_their_storage(bucket, tmp_path, capsys):
    _bucket_with(
        bucket, {"fra/00000": [(b"k", "fra_Latn", "salut")], "fra/00001": [(b"j", "fra_Latn", "oui")]}
    )
    bucket.data["claims/fra/00001.1000"] = b""
    run = workqueue.Worker(bucket, _upper, tmp_path, 0, budget_s=1e6, clock=Clock(1.0))
    assert run.run() == 1
    assert sorted(bucket.data) == [
        "claims/fra/00001.1000",
        "results/fra/00000.parquet",
        "work/fra/00001.parquet",
    ]
    assert pl.read_parquet(io.BytesIO(bucket.data["results/fra/00000.parquet"])).rows() == [(b"k", "SALUT")]
    claim = next(c for c in bucket.calls if c[0] == "put")[1][0]
    assert claim.startswith("claims/fra/00000.")
    assert ("delete", ["work/fra/00000.parquet", claim]) in bucket.calls
    assert "fra/00000: 1 texts in" in capsys.readouterr().out


def test_worker_skips_shards_with_results_and_stops_when_time_is_short(bucket, tmp_path):
    _bucket_with(
        bucket,
        {
            "a/00000": [(b"k", "fra_Latn", "x")],
            "a/00001": [(b"j", "fra_Latn", "y")],
            "a/00002": [(b"i", "fra_Latn", "z")],
        },
    )
    bucket.data["results/a/00000.parquet"] = b""
    run = workqueue.Worker(bucket, _upper, tmp_path, 0, budget_s=25, clock=Clock(5.0))
    assert run.run() == 1
    assert "results/a/00001.parquet" in bucket.data
    assert "work/a/00002.parquet" in bucket.data


def test_shard_name():
    assert workqueue.shard_name("work/fra_Latn/00003.parquet") == "fra_Latn/00003"


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_worker_with_no_time_does_nothing(bucket, tmp_path, budget):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")]})
    assert workqueue.Worker(bucket, _upper, tmp_path, 0, budget_s=budget, clock=Clock(0.0)).run() == 0


class RecordingBucket:
    """Wraps the shared fake bucket and keeps the bytes of every upload."""

    def __init__(self, bucket):
        self.inner, self.uploads = bucket, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def put(self, files):
        self.uploads += [(remote, local if isinstance(local, bytes) else None) for local, remote in files]
        self.inner.put(files)


def test_process_claims_with_an_empty_file_keeps_two_scratch_files_and_reports(bucket, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    _bucket_with(bucket, {"fra/00000": [(b"k", "fra_Latn", "salut"), (b"j", "fra_Latn", "oui")]})
    printed = []
    monkeypatch.setattr(workqueue, "print", lambda *a, **k: printed.append((a, k)), raising=False)
    recording = RecordingBucket(bucket)
    run = workqueue.Worker(recording, _upper, tmp_path, 0, budget_s=1e6, clock=Clock(1.0))
    run.process("fra/00000", 1000)
    assert recording.uploads == [("claims/fra/00000.1000", b""), ("results/fra/00000.parquet", None)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.parquet", "work.parquet"]
    meta = pq.ParquetFile(tmp_path / "result.parquet").metadata
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert printed == [(("fra/00000: 2 texts in 1 s",), {"flush": True})]


def test_results_paths_map_back_to_their_work_shard_only_once(bucket):
    bucket.data.update({"work/x/results/0.parquet": b"", "results/x/results/0.parquet": b""})
    run = workqueue.Worker(bucket, _upper, None, 0, budget_s=1e6, clock=Clock(0.0))
    assert run.next_shard() is None


def test_run_reserves_no_time_before_the_first_shard(bucket, tmp_path):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")], "a/00001": [(b"j", "fra_Latn", "y")]})
    run = workqueue.Worker(bucket, _upper, tmp_path, 0, budget_s=1.5, clock=Clock(1.0))
    assert run.run() == 1


def test_run_measures_each_shard_by_its_own_duration(bucket, tmp_path):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")], "a/00001": [(b"j", "fra_Latn", "y")]})
    run = workqueue.Worker(bucket, _upper, tmp_path, 0, budget_s=100, clock=Clock(1.0))
    assert run.run() == 2


def test_worker_writes_arrow_tables_as_they_are(bucket, tmp_path):
    import pyarrow as pa

    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")]})
    run = workqueue.Worker(
        bucket, lambda work: pa.table({"n": [work.height]}), tmp_path, 0, budget_s=1e6, clock=Clock(1.0)
    )
    assert run.run() == 1
    assert pl.read_parquet(io.BytesIO(bucket.data["results/a/00000.parquet"])).rows() == [(1,)]
