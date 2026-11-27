import polars as pl

from georeset.label import jobs

from .fakes import FakeBucket, FakeEngine, completion


def encode(prompts):
    """Stand-in tokenizer: the i-th prompt becomes the single token id i."""
    return [[i] for i in range(len(prompts))]


def sample(landuse=(True, True, False, False)):
    return pl.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text": ["Forest.", "Forest.", "Born in 1900.", "123"],
            "landuse": list(landuse),
        }
    )


def test_plan_work_report_agree_with_the_labels(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket")
    jobs.plan(sample(), bucket, tmp_path / "plan", encode)
    engine = FakeEngine({(0,): completion("</think>yes"), (1,): completion("</think>no")})
    jobs.work(bucket, tmp_path / "work", engine)
    assert jobs.report(bucket, tmp_path / "report") == {
        "rows": 4,
        "matches": 4,
        "failed": 0,
        "mismatches": [],
    }
    # Each distinct text is planned once, and the letterless one never reaches the model.
    assert engine.calls == [(0,), (1,)]


def test_report_lists_the_rows_whose_label_differs(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket")
    jobs.plan(sample(landuse=(True, True, True, False)), bucket, tmp_path / "plan", encode)
    engine = FakeEngine({(0,): completion("</think>yes"), (1,): completion("</think>no")})
    jobs.work(bucket, tmp_path / "work", engine)
    assert jobs.report(bucket, tmp_path / "report") == {
        "rows": 4,
        "matches": 3,
        "failed": 0,
        "mismatches": [{"id": "3", "decision": "no", "landuse": True}],
    }


def test_unparsed_answers_are_counted_as_failed(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket")
    jobs.plan(sample(), bucket, tmp_path / "plan", encode)
    engine = FakeEngine({(0,): completion("</think>yes"), (1,): completion("</think>maybe")})
    jobs.work(bucket, tmp_path / "work", engine)
    summary = jobs.report(bucket, tmp_path / "report")
    assert summary["failed"] == 1
    assert summary["mismatches"] == [{"id": "3", "decision": "failed", "landuse": False}]


def test_remove_deletes_the_whole_bucket():
    class Bucket:
        removed = False

        def remove(self):
            self.removed = True

    bucket = Bucket()
    jobs.remove(bucket)
    assert bucket.removed
