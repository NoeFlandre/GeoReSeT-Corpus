import polars as pl
import pytest

from georeset.label import migrate


def write(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)
    return path


def test_strip_label_drops_only_the_landuse_column(tmp_path):
    src, dst = tmp_path / "src.parquet", tmp_path / "out" / "dst.parquet"
    write(src, {"id": ["a", "b"], "text": ["x", "y"], "landuse": [True, False], "source": ["w", "w"]})
    rows = migrate.strip_label(src, dst)
    out = pl.read_parquet(dst)
    assert rows == 2
    assert out.columns == ["id", "text", "source"]
    assert out["text"].to_list() == ["x", "y"]


def test_rows_and_inline_labels_count_across_files(tmp_path):
    labelled = write(tmp_path / "a.parquet", {"id": ["a", "b"], "landuse": [True, False]})
    bare = write(tmp_path / "b.parquet", {"id": ["c"]})
    assert migrate.rows([labelled, bare]) == 3
    assert migrate.inline_labels([labelled, bare]) == 1


def test_check_migrated_accepts_a_complete_migration():
    migrate.check_migrated(data_rows=5, decision_rows=5, inline=0)


def test_check_migrated_refuses_an_inline_label_left_behind():
    with pytest.raises(ValueError, match="still have the inline landuse column"):
        migrate.check_migrated(data_rows=5, decision_rows=5, inline=2)


def test_check_migrated_refuses_a_row_count_mismatch():
    with pytest.raises(ValueError, match="4 decisions for 5 data rows"):
        migrate.check_migrated(data_rows=5, decision_rows=4, inline=0)
