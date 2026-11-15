import io

import polars as pl
import pytest

from georeset.translate import worker
from georeset.translate.worker import Output


class FakeEngine:
    """Counts one token per word; translates by upper-casing; records each call's budgets."""

    def __init__(self):
        self.calls = []
        self.counted = []

    def count(self, texts, language):
        self.counted.append(language)
        return [len(t.split()) for t in texts]

    def translate(self, texts, language, budgets):
        self.calls.append((language, list(budgets)))
        return [Output(f"en {t}", min(len(t.split()) + 1, b)) for t, b in zip(texts, budgets, strict=True)]


def _english(texts):
    return ["eng_Latn"] * len(texts)


def _work(rows):
    return pl.DataFrame(
        rows, schema={"key": pl.Binary, "translation_language": pl.String, "text": pl.String}, orient="row"
    )


def test_flags_name_the_failed_checks():
    assert worker.flags("Bonjour le monde", Output("Hello world", 3), 32, 3, "eng_Latn") == []
    assert worker.flags("Bonjour", Output("", 1), 32, 1, "") == ["empty", "length_ratio_outlier"]
    assert worker.flags("Bonjour le monde", Output("Bonjour le monde", 4), 32, 3, "fra_Latn") == [
        "untranslated_copy",
        "wrong_language",
    ]
    assert worker.flags("a b c", Output("x x x x x x x x x x", 32), 32, 3, "eng_Latn") == [
        "repetition_loop",
        "length_ratio_outlier",
        "truncated",
    ]
    assert worker.flags("a", Output("b", 31), 32, 1, "eng_Latn") == []
    assert worker.flags("a", Output("b", 2), 32, worker.MAX_INPUT_TOKENS - 1, "eng_Latn") == [
        "input_truncated"
    ]
    assert worker.flags("a", Output("b", 2), 32, worker.MAX_INPUT_TOKENS - 2, "eng_Latn") == []


def test_translate_group_gives_each_chunk_its_own_budgets(monkeypatch):
    monkeypatch.setattr(worker, "CHUNK", 2)
    engine = FakeEngine()
    texts = ["a", "b c", " ".join(["w"] * 20), " ".join(["v"] * 340), " ".join(["u"] * 341)]
    outputs, budgets, counts = worker.translate_group(engine, texts, "fra_Latn")
    assert counts == [1, 2, 20, 340, 341]
    assert budgets == [32, 32, 60, 1020, worker.MAX_OUTPUT_TOKENS]
    assert worker.MAX_OUTPUT_TOKENS == 1020
    assert engine.calls == [("fra_Latn", [32, 32]), ("fra_Latn", [60, 1020]), ("fra_Latn", [1020])]
    assert engine.counted == ["fra_Latn"]
    assert [o.text for o in outputs][:3] == ["en a", "en b c", "en " + " ".join(["w"] * 20)]


def test_translate_shard_groups_by_language_and_keeps_keys():
    engine = FakeEngine()
    work = _work([(b"k1", "deu_Latn", "hallo welt"), (b"k2", "fra_Latn", "salut"), (b"k3", "deu_Latn", "ja")])
    out = worker.translate_shard(engine, _english, work)
    assert out.rows() == [(b"k1", "en hallo welt", []), (b"k3", "en ja", []), (b"k2", "en salut", [])]
    assert out.schema == {"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)}
    assert [language for language, _ in engine.calls] == ["deu_Latn", "fra_Latn"]


def test_claims_keep_recent_ones():
    paths = ["claims/fra_Latn/00001.1000", "claims/fra_Latn/00002.100", "claims/a.b/00003.999"]
    assert worker.claims(paths, now=1000 + worker.CLAIM_S - 1) == {"fra_Latn/00001"}
    assert worker.claims(paths, now=1000.0) == {"fra_Latn/00001", "fra_Latn/00002", "a.b/00003"}


def test_choose_skips_taken_shards_and_spreads_by_offset():
    work = ["work/a/00000.parquet", "work/a/00001.parquet", "work/b/00000.parquet"]
    assert worker.choose(work, {"a/00001"}, 0) == "a/00000"
    assert worker.choose(work, {"a/00001"}, 3) == "b/00000"
    assert worker.choose(work, {"a/00000", "a/00001", "b/00000"}, 0) is None


def test_may_start_only_if_the_longest_shard_still_fits():
    assert worker.may_start(10, 20, 31)
    assert not worker.may_start(10, 20, 30)


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
    run = worker.Worker(bucket, FakeEngine(), _english, tmp_path, 0, budget_s=1e6, clock=Clock(1.0))
    assert run.run() == 1
    assert sorted(bucket.data) == [
        "claims/fra/00001.1000",
        "results/fra/00000.parquet",
        "work/fra/00001.parquet",
    ]
    assert pl.read_parquet(io.BytesIO(bucket.data["results/fra/00000.parquet"])).rows() == [
        (b"k", "en salut", [])
    ]
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
    run = worker.Worker(bucket, FakeEngine(), _english, tmp_path, 0, budget_s=25, clock=Clock(5.0))
    assert run.run() == 1
    assert "results/a/00001.parquet" in bucket.data
    assert "work/a/00002.parquet" in bucket.data


def test_shard_name():
    assert worker.shard_name("work/fra_Latn/00003.parquet") == "fra_Latn/00003"


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_worker_with_no_time_does_nothing(bucket, tmp_path, budget):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")]})
    assert (
        worker.Worker(bucket, FakeEngine(), _english, tmp_path, 0, budget_s=budget, clock=Clock(0.0)).run()
        == 0
    )


def test_translate_shard_refuses_a_language_label_per_output_mismatch():
    work = _work([(b"k1", "deu_Latn", "hallo"), (b"k2", "deu_Latn", "welt")])
    with pytest.raises(ValueError, match="zip"):
        worker.translate_shard(FakeEngine(), lambda texts: ["eng_Latn"], work)


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
    monkeypatch.setattr(worker, "print", lambda *a, **k: printed.append((a, k)), raising=False)
    recording = RecordingBucket(bucket)
    run = worker.Worker(recording, FakeEngine(), _english, tmp_path, 0, budget_s=1e6, clock=Clock(1.0))
    run.process("fra/00000", 1000)
    assert recording.uploads == [("claims/fra/00000.1000", b""), ("results/fra/00000.parquet", None)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.parquet", "work.parquet"]
    meta = pq.ParquetFile(tmp_path / "result.parquet").metadata
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert printed == [(("fra/00000: 2 texts in 1 s",), {"flush": True})]


def test_results_paths_map_back_to_their_work_shard_only_once(bucket):
    bucket.data.update({"work/x/results/0.parquet": b"", "results/x/results/0.parquet": b""})
    run = worker.Worker(bucket, FakeEngine(), _english, None, 0, budget_s=1e6, clock=Clock(0.0))
    assert run.next_shard() is None


def test_run_reserves_no_time_before_the_first_shard(bucket, tmp_path):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")], "a/00001": [(b"j", "fra_Latn", "y")]})
    run = worker.Worker(bucket, FakeEngine(), _english, tmp_path, 0, budget_s=1.5, clock=Clock(1.0))
    assert run.run() == 1


def test_run_measures_each_shard_by_its_own_duration(bucket, tmp_path):
    _bucket_with(bucket, {"a/00000": [(b"k", "fra_Latn", "x")], "a/00001": [(b"j", "fra_Latn", "y")]})
    run = worker.Worker(bucket, FakeEngine(), _english, tmp_path, 0, budget_s=100, clock=Clock(1.0))
    assert run.run() == 2
