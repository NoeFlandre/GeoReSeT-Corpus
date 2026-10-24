import json
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
    target = dest / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(path)
    return target


def test_run_task_downloads_available_files_only(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "download", _fake_download)
    task = stage.Task(SOURCE, "r", "rev", frozenset({"a/r"}))
    out = stage.run_task(task, tmp_path, threading.Semaphore(1))
    assert out == tmp_path / "stage" / "fake" / "r"
    assert pl.read_parquet(out / "links.parquet")["files"].to_list() == ["a"]
    assert sorted(p.name for p in out.iterdir()) == ["links.parquet", "polygons.parquet", "sentences.parquet"]
    assert list((tmp_path / "tmp").iterdir()) == []


def test_run_task_skips_done_region(tmp_path, monkeypatch):
    done = stage.stage_dir(tmp_path, "fake", "r")
    done.mkdir(parents=True)
    monkeypatch.setattr(hub, "download", lambda *a: (_ for _ in ()).throw(AssertionError("no download")))
    assert (
        stage.run_task(stage.Task(SOURCE, "r", "rev", frozenset()), tmp_path, threading.Semaphore(1)) == done
    )


def test_tasks_and_stage_record_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "revision", lambda repo: "rev1")
    monkeypatch.setattr(hub, "list_files", lambda repo, rev: frozenset({"a/r1", "b/r1", "a/r2"}))
    monkeypatch.setattr(hub, "download", _fake_download)
    todo = stage.tasks([SOURCE], ["r1", "r2"])
    assert [(t.stem, t.rev) for t in todo] == [("r1", "rev1"), ("r2", "rev1")]
    outs = stage.stage(todo, tmp_path, downloads=2, transforms=1)
    assert [pl.read_parquet(o / "links.parquet")["files"].item() for o in outs] == ["a,b", "a"]
    assert json.loads((tmp_path / "revisions.json").read_text()) == {"me/fake": "rev1"}
