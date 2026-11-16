import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from georeset.translate import assemble, route


def _table(status=False):
    columns = {
        "id": ["1", "2", "3", "4"],
        "text": ["Bonjour", "Hello", "Salve", "Hallo"],
        "landuse": [True, False, True, True],
        "language": ["fra_Latn", "eng_Latn", "lat_Latn", "eng_Latn"],
        "source": ["wikipedia", "website", "wikipedia", "website"],
        "polygon_ids": [["way/1"], ["way/2"], [], ["way/3"]],
    }
    if status:
        columns["status"] = ["failed"] * 4
    table = pa.table(columns)
    return table.replace_schema_metadata({b"k": b"v"})


OVERRIDES = pl.DataFrame({"id": ["4"], "detected": ["deu_Latn"]})
TRANSLATIONS = pl.DataFrame(
    {
        "key": [route.key("fra_Latn", "Bonjour"), route.key("deu_Latn", "Hallo"), b"other"],
        "text_en": ["Hello", "Hello", "x"],
        "translation_flags": [[], ["truncated"], []],
    },
    schema=assemble.TRANSLATIONS_SCHEMA,
)


def test_translation_columns():
    out = assemble.translation_columns(pl.DataFrame(_table()), OVERRIDES, TRANSLATIONS)
    assert out.columns == list(assemble.NEW_COLUMNS)
    assert out.rows() == [
        ("Hello", "nllb", "fra_Latn", []),
        ("Hello", "original", None, []),
        (None, "unsupported", None, []),
        ("Hello", "nllb", "deu_Latn", ["truncated"]),
    ]


def test_a_routed_row_without_translation_stops_everything():
    with pytest.raises(assemble.MissingTranslation, match="2 rows"):
        assemble.translation_columns(pl.DataFrame(_table()), OVERRIDES, TRANSLATIONS.head(0))


def test_rewrite_appends_the_columns_and_keeps_the_rest(tmp_path):
    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(_table(status=True), path, row_group_size=3, compression="zstd")
    assemble.rewrite(path, OVERRIDES, TRANSLATIONS)
    new = pq.read_table(path)
    assert new.column_names == [*_table(status=True).column_names, *assemble.NEW_COLUMNS]
    assert new.select(_table(status=True).column_names).equals(_table(status=True))
    assert new.schema.metadata == {b"k": b"v"}
    meta = pq.ParquetFile(path).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [3, 1]
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert new.column("text_en").to_pylist() == ["Hello", "Hello", None, "Hello"]
    assert not path.with_name(path.name + ".tmp").exists()


def test_rewrite_of_an_empty_shard(tmp_path):
    path = tmp_path / "s.parquet"
    pq.write_table(_table().slice(0, 0), path)
    assemble.rewrite(path, OVERRIDES, TRANSLATIONS)
    assert pq.read_table(path).num_rows == 0


def test_rewrite_refuses_a_file_whose_originals_changed(tmp_path, monkeypatch):
    path = tmp_path / "s.parquet"
    pq.write_table(_table(), path)
    before = path.read_bytes()
    monkeypatch.setattr(assemble, "same_originals", lambda old, new: False)
    with pytest.raises(RuntimeError, match="original columns changed"):
        assemble.rewrite(path, OVERRIDES, TRANSLATIONS)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.parquet"]


def test_same_originals():
    old = pa.table({"a": [1, 2]})
    good = old.append_column("text_en", pa.array(["x", "y"]))
    for name in assemble.NEW_COLUMNS[1:]:
        good = good.append_column(name, pa.array(["x", "y"]))
    assert assemble.same_originals(old, good)
    assert not assemble.same_originals(old, good.set_column(0, "a", pa.array([1, 3])))
    assert not assemble.same_originals(old, good.drop_columns(["translation"]))
    assert not assemble.same_originals(pa.table({"a": [1]}), pa.table({"a": [1, 1]}).select(["a"]))


def test_with_translations_keeps_metadata():
    out = assemble.with_translations(_table(), pl.DataFrame({"x": [1, 2, 3, 4]}))
    assert out.column_names[-1] == "x"
    assert out.schema.metadata == {b"k": b"v"}


def test_load_translations_deduplicates(tmp_path):
    TRANSLATIONS.write_parquet(tmp_path / "a.parquet")
    TRANSLATIONS.head(1).write_parquet(tmp_path / "b.parquet")
    assert assemble.load_translations([tmp_path / "a.parquet", tmp_path / "b.parquet"]).equals(TRANSLATIONS)
    assert assemble.load_translations([]).schema == assemble.TRANSLATIONS_SCHEMA


def test_translation_columns_keep_unsplit_rows_unsplit():
    frame = pl.DataFrame(_table(status=True)).with_columns(status=pl.lit("unsplit"))
    out = assemble.translation_columns(frame, OVERRIDES, TRANSLATIONS)
    assert out["translation"].to_list() == ["unsplit", "original", "unsplit", "unsplit"]


def test_translation_columns_keep_the_rows_order():
    n = 300
    texts = [f"Bonjour {i % 40}" for i in range(n)]
    frame = pl.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "text": texts,
            "language": ["fra_Latn"] * n,
            "source": ["wikipedia"] * n,
        }
    )
    translations = pl.DataFrame(
        {
            "key": [route.key("fra_Latn", f"Bonjour {i}") for i in reversed(range(40))],
            "text_en": [f"Hello {i}" for i in reversed(range(40))],
            "translation_flags": [[]] * 40,
        },
        schema=assemble.TRANSLATIONS_SCHEMA,
    )
    out = assemble.translation_columns(frame, OVERRIDES.head(0), translations)
    assert out["text_en"].to_list() == [t.replace("Bonjour", "Hello") for t in texts]


def test_load_translations_keep_the_first_result_of_each_key_in_order(tmp_path):
    n = 200
    first = pl.DataFrame(
        {"key": [str(i).encode() for i in range(n)], "text_en": ["a"] * n, "translation_flags": [[]] * n},
        schema=assemble.TRANSLATIONS_SCHEMA,
    )
    first.write_parquet(tmp_path / "a.parquet")
    first.with_columns(text_en=pl.lit("b")).write_parquet(tmp_path / "b.parquet")
    assert assemble.load_translations([tmp_path / "a.parquet", tmp_path / "b.parquet"]).equals(first)


def test_translation_flags_are_strings_even_when_no_result_has_any():
    untyped = pl.DataFrame(
        {"key": [route.key("fra_Latn", "Bonjour")], "text_en": ["Hello"], "translation_flags": [[]]}
    )
    assert untyped.schema["translation_flags"] == pl.List(pl.Null)
    frame = pl.DataFrame(_table()).head(3)
    out = assemble.translation_columns(frame, OVERRIDES, untyped)
    assert out.schema["translation_flags"] == pl.List(pl.String)
