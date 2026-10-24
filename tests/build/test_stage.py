import threading

import polars as pl

from georeset import hub
from georeset.build import stage
from georeset.sources.base import Source, Staged


def _transform(paths):
    names = sorted(paths)
    frame = pl.DataFrame({"x": [1]})
    return Staged(polygons=frame, links=pl.DataFrame({"files": [",".join(names)]}), sentences=frame)


SOURCE = Source("fake", "me/fake", lambda stem: {"a": f"a/{stem}", "b": f"b/{stem}"}, _transform)


def _fake_download(repo, path, rev, dest):
    assert (repo, rev) == ("me/fake", "rev1"), (repo, rev)
    target = dest / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(path)
    return target


def test_run_task_downloads_available_files_only(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "download", _fake_download)
    dests = []
    monkeypatch.setattr(hub, "download", lambda *a: dests.append(a[3]) or _fake_download(*a))
    task = stage.Task(SOURCE, "r", "rev1", frozenset({"a/r"}))
    out = stage.run_task(task, tmp_path, threading.Semaphore(1))
    assert out == tmp_path / "stage" / "fake" / "r"
    assert pl.read_parquet(out / "links.parquet")["files"].to_list() == ["a"]
    assert sorted(p.name for p in out.iterdir()) == ["links.parquet", "polygons.parquet", "sentences.parquet"]
    assert list((tmp_path / "tmp").iterdir()) == []
    assert [d.parent for d in dests] == [tmp_path / "tmp"]


def test_run_task_skips_done_region(tmp_path, monkeypatch):
    done = stage.stage_dir(tmp_path, "fake", "r")
    done.mkdir(parents=True)
    monkeypatch.setattr(hub, "download", lambda *a: (_ for _ in ()).throw(AssertionError("no download")))
    assert (
        stage.run_task(stage.Task(SOURCE, "r", "rev", frozenset()), tmp_path, threading.Semaphore(1)) == done
    )


OTHER = Source("other", "me/other", lambda stem: {"a": f"a/{stem}"}, _transform)
REVISIONS = {"me/fake": "rev1", "me/other": "rev2"}


def test_tasks_pin_one_revision_per_source(monkeypatch):
    listed = []
    monkeypatch.setattr(hub, "revision", REVISIONS.__getitem__)
    monkeypatch.setattr(hub, "list_files", lambda repo, rev: listed.append((repo, rev)) or frozenset({repo}))
    todo = stage.tasks([SOURCE, OTHER], ["r1", "r2"])
    assert [(t.source.name, t.stem, t.rev, t.available) for t in todo] == [
        ("fake", "r1", "rev1", frozenset({"me/fake"})),
        ("fake", "r2", "rev1", frozenset({"me/fake"})),
        ("other", "r1", "rev2", frozenset({"me/other"})),
        ("other", "r2", "rev2", frozenset({"me/other"})),
    ]
    assert listed == [("me/fake", "rev1"), ("me/other", "rev2")]


def test_stage_runs_tasks_and_records_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "download", _fake_download)
    todo = [
        stage.Task(SOURCE, "r1", "rev1", frozenset({"a/r1", "b/r1"})),
        stage.Task(SOURCE, "r2", "rev1", frozenset({"a/r2"})),
    ]
    outs = stage.stage(todo, tmp_path / "work", downloads=2, transforms=1)
    assert [pl.read_parquet(o / "links.parquet")["files"].item() for o in outs] == ["a,b", "a"]
    assert (tmp_path / "work" / "revisions.json").read_text() == '{\n  "me/fake": "rev1"\n}'


def test_write_staged_replaces_a_stale_tmp(tmp_path):
    out = tmp_path / "stage" / "s" / "r"
    stale = out.with_name("r.tmp")
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    frame = pl.DataFrame({"x": [1]})
    stage.write_staged(Staged(polygons=frame, links=frame, sentences=frame), out)
    assert sorted(p.name for p in out.iterdir()) == ["links.parquet", "polygons.parquet", "sentences.parquet"]
    assert not stale.exists()


def test_conversion_runs_inside_the_semaphore_but_downloads_do_not(tmp_path, monkeypatch):
    events = []

    class Gate:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *a):
            events.append("exit")

    def download(*a):
        events.append("download")
        return _fake_download(*a)

    def transform(paths):
        events.append("transform")
        return _transform(paths)

    monkeypatch.setattr(hub, "download", download)
    source = Source("fake", "me/fake", SOURCE.files, transform)
    stage.run_task(stage.Task(source, "r", "rev1", frozenset({"a/r"})), tmp_path, Gate())
    assert events == ["download", "enter", "transform", "exit"]
