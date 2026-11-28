from pathlib import Path

import polars as pl

from georeset.label import attach


def _round_folder(out: Path) -> Path:
    folder = out / "labels" / attach.CURRENT
    folder.mkdir(parents=True)
    pl.DataFrame(
        {"id": ["a", "b"], "text_sha256": ["x", "y"], "decision": ["yes", "no"], "failure": [None, None]}
    ).write_parquet(folder / "decisions.parquet")
    return out


def test_landuse_is_the_release_column_when_it_has_one(tmp_path):
    data = pl.LazyFrame({"id": ["a"], "landuse": [False]})
    assert attach.landuse(data, tmp_path).collect()["landuse"].to_list() == [False]


def test_landuse_is_the_round_decision_when_the_release_has_no_column(tmp_path):
    out = _round_folder(tmp_path)
    data = pl.LazyFrame({"id": ["a", "b"], "text": ["t1", "t2"]})
    joined = attach.landuse(data, out).sort("id").collect()
    assert joined["landuse"].to_list() == [True, False]
    assert joined["text"].to_list() == ["t1", "t2"]
