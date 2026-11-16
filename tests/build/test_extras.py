import json
from pathlib import Path

import polars as pl
import pytest

from georeset.build import extras

CARD = "7 sentences were translated into English with NLLB (1,234 distinct texts from 3 source languages)."


def test_translation_summary_takes_the_count_from_the_card_and_the_settings_from_the_code():
    assert extras.translation_summary(CARD) == {
        "distinct": 1234,
        "model": "facebook/nllb-200-3.3B",
        "revision": extras.models.NLLB_REVISION,
        "threshold": 0.9,
    }


def test_translation_summary_needs_exactly_one_count():
    with pytest.raises(ValueError):
        extras.translation_summary("no translation here")


def test_fetch_pins_the_release_and_the_fused_folders(tmp_path, monkeypatch):
    calls = []
    out, work = tmp_path / "out", tmp_path / "work"

    def fetch_corpus(repo, rev, o, w):
        calls.append(("corpus", repo, rev, o, w))
        (w / "corpus").mkdir(parents=True)
        (w / "corpus" / "README.md").write_text(CARD)

    monkeypatch.setattr(extras.fuse, "fetch_corpus", fetch_corpus)
    monkeypatch.setattr(extras.hub, "revision_of", lambda repo, ref: f"{repo}@{ref}")
    monkeypatch.setattr(extras.hub, "revision", {extras.FUSED_REPO: "f" * 40}.get)
    monkeypatch.setattr(extras.hub, "snapshot", lambda *a: calls.append(("snapshot", *a)))
    extras.fetch(out, work)
    assert calls == [
        ("corpus", extras.REPO, f"{extras.REPO}@v1.1", out, work),
        ("snapshot", extras.FUSED_REPO, "f" * 40, out, ["embeddings/*", "polygons_labels_shares/*"]),
    ]
    assert sorted(p.name for p in work.iterdir()) == [
        "corpus",
        "fused.json",
        "revisions.json",
        "translation.json",
    ]
    fused = {"NoeFlandre/GeoReSeT-Corpus-Fused": "f" * 40}
    assert (work / "fused.json").read_text() == json.dumps(fused, indent=2)
    summary = extras.translation_summary(CARD)
    assert (work / "translation.json").read_text() == json.dumps(summary, indent=2)


SHA = "a" * 40
PINNED_CARD = (
    f"{CARD}\n* [`me/up` @ `aaaaaaa`](https://huggingface.co/datasets/me/up/tree/{SHA})\n"
    f"Copied from [x](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus-Fused/tree/{'b' * 40})"
)


def test_restore_context_splits_the_fused_pin_from_the_label_pins(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "README.md").write_text(PINNED_CARD)
    extras.restore_context(tmp_path)
    assert (tmp_path / "revisions.json").read_text() == json.dumps({"me/up": SHA}, indent=2)
    fused = {extras.FUSED_REPO: "b" * 40}
    assert (tmp_path / "fused.json").read_text() == json.dumps(fused, indent=2)
    summary = extras.translation_summary(CARD)
    assert (tmp_path / "translation.json").read_text() == json.dumps(summary, indent=2)


def test_restore_context_without_a_fused_pin_writes_none(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "README.md").write_text(CARD)
    extras.restore_context(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus", "revisions.json", "translation.json"]
    assert json.loads((tmp_path / "revisions.json").read_text()) == {}


def test_fetch_reads_the_release_card_by_its_exact_name(tmp_path, monkeypatch):
    def fetch_corpus(repo, rev, o, w):
        (w / "corpus").mkdir(parents=True)
        (w / "corpus" / "README.md").write_text(CARD)

    monkeypatch.setattr(extras.fuse, "fetch_corpus", fetch_corpus)
    monkeypatch.setattr(extras.hub, "revision_of", lambda repo, ref: "r")
    monkeypatch.setattr(extras.hub, "revision", lambda repo: "f")
    monkeypatch.setattr(extras.hub, "snapshot", lambda *a: None)
    read, read_text = [], Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self: read.append(self) or read_text(self))
    extras.fetch(tmp_path / "out", tmp_path / "work")
    assert [p.parts[-2:] for p in read] == [("corpus", "README.md")]


def _write(folder, df, name="train-0.parquet"):
    folder.mkdir(parents=True, exist_ok=True)
    df.write_parquet(folder / name)


def _release(out, *, vectors=None, shares=None):
    """Sentences a, b (same text as a) and c relevant, d not; polygons way/1 and way/2."""
    _write(
        out / "data" / "wikipedia" / "fra_Latn",
        pl.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "text": ["x", "x", "y", "z"],
                "landuse": [True, True, True, False],
                "lat": [1.0, 1.0, 2.0, 3.0],
                "lon": [4.0, 4.0, 5.0, 6.0],
            }
        ),
    )
    _write(out / "polygons", pl.DataFrame({"polygon_id": ["way/1", "way/2"]}))
    default = {"id": ["a", "c"], "lat": [1.0, 2.0], "lon": [4.0, 5.0]}
    _write(out / "embeddings", pl.DataFrame(vectors or default))
    _write(out / "polygons_labels_shares", pl.DataFrame(shares or {"polygon_id": ["way/1", "way/2"]}))
    return out


def test_check_passes_when_the_folders_fit_the_release(tmp_path):
    assert extras.check(_release(tmp_path)) == []


@pytest.mark.parametrize(
    ("vectors", "shares", "problem"),
    [
        (
            {"id": ["a", "a", "c"], "lat": [1.0, 1.0, 2.0], "lon": [4.0, 4.0, 5.0]},
            None,
            "embeddings sharing an id: 1",
        ),
        (
            {"id": ["a", "c", "d"], "lat": [1.0, 2.0, 3.0], "lon": [4.0, 5.0, 6.0]},
            None,
            "of no relevant sentence: 1",
        ),
        (
            {"id": ["a", "c"], "lat": [1.0, 2.0], "lon": [4.0, 9.0]},
            None,
            "embeddings away from their sentence: 1",
        ),
        (
            {"id": ["a", "c"], "lat": [1.0, 7.0], "lon": [4.0, 5.0]},
            None,
            "embeddings away from their sentence: 1",
        ),
        ({"id": ["a"], "lat": [1.0], "lon": [4.0]}, None, "relevant texts without an embedding: 1"),
        (None, {"polygon_id": ["way/1", "way/1", "way/2"]}, "label shares sharing a polygon: 1"),
        (None, {"polygon_id": ["way/1", "way/2", "way/3"]}, "label shares of no polygon: 1"),
        (None, {"polygon_id": ["way/1"]}, "polygons without label shares: 1"),
    ],
    ids=["dup-id", "not-relevant", "moved-lon", "moved-lat", "missing", "dup-polygon", "extra", "uncovered"],
)
def test_check_reports_what_does_not_fit(tmp_path, vectors, shares, problem):
    (found,) = extras.check(_release(tmp_path, vectors=vectors, shares=shares))
    assert found.endswith(problem)


def test_check_scans_each_folder_by_its_exact_name(tmp_path, monkeypatch):
    out = _release(tmp_path)
    scanned, scan = [], pl.scan_parquet
    monkeypatch.setattr(extras.pl, "scan_parquet", lambda path: scanned.append(path) or scan(path))
    extras.check(out)
    assert {str(p.relative_to(out)) for p in scanned} == {
        "data/*/*/*.parquet",
        "embeddings/*.parquet",
        "polygons_labels_shares/*.parquet",
        "polygons/*.parquet",
    }


def test_check_counts_with_thousands_separators(tmp_path, monkeypatch):
    monkeypatch.setattr(
        extras, "_checks", lambda out: {"x": pl.LazyFrame({"n": [1234]}), "y": pl.LazyFrame({"n": [0]})}
    )
    assert extras.check(tmp_path) == ["x: 1,234"]


def test_assemble_refuses_folders_that_do_not_fit(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(extras, "fetch", lambda out, work: fetched.append((out, work)))
    checked = []
    monkeypatch.setattr(extras, "check", lambda out: checked.append(out) or ["a: 1", "b: 2"])
    with pytest.raises(ValueError, match=r"^a: 1; b: 2$"):
        extras.assemble(tmp_path / "out", tmp_path / "work")
    assert fetched == [(tmp_path / "out", tmp_path / "work")]
    assert checked == [tmp_path / "out"]
    monkeypatch.setattr(extras, "check", lambda out: [])
    extras.assemble(tmp_path / "out", tmp_path / "work")


def test_propose_leaves_the_release_files_out(tmp_path, monkeypatch):
    for folder in (
        "data/wikipedia",
        "unlabelled",
        "polygons",  # no assets: a missing folder is skipped
        "embeddings",
        "polygons_labels_shares",
    ):
        (tmp_path / folder).mkdir(parents=True)
    seen = []
    monkeypatch.setattr(
        extras.hub,
        "propose",
        lambda out, repo, title: seen.append((sorted(p.name for p in out.iterdir()), repo, title)) or "url",
    )
    assert extras.propose(tmp_path) == "url"
    assert seen == [(["embeddings", "polygons_labels_shares"], extras.REPO, extras.PR_TITLE)]


def test_finalize_merges_then_tags_v1_2(monkeypatch):
    calls = []
    monkeypatch.setattr(extras.hub, "merge", lambda repo, n: calls.append(("merge", repo, n)))
    monkeypatch.setattr(extras.hub, "tag", lambda repo, name: calls.append(("tag", repo, name)))
    extras.finalize(5)
    assert calls == [("merge", extras.REPO, 5), ("tag", extras.REPO, "v1.2")]
