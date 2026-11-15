import json

import polars as pl

from georeset.translate import plan, route

SCHEMA = {"id": pl.String, "text": pl.String, "language": pl.String, "source": pl.String}


def _shard(path, rows, status=None):
    frame = pl.DataFrame(rows, schema=SCHEMA, orient="row").with_columns(landuse=pl.lit(True))
    if status:
        frame = frame.with_columns(status=pl.lit(status))
    frame.write_parquet(path)
    return path


def _identify(texts):
    labels = {"Home": ("eng_Latn", 0.99), "Hallo Welt": ("deu_Latn", 0.95)}
    found = [labels.get(t, ("hin_Deva", 0.5)) for t in texts]
    return [label for label, _ in found], [p for _, p in found]


def test_units_are_language_folders_then_unlabelled():
    paths = [
        "data/wikipedia/fra_Latn/train-0.parquet",
        "data/website/fra_Latn/train-0.parquet",
        "data/website/deu_Latn/train-0.parquet",
        "data/x.parquet",
        "polygons/train-0.parquet",
        "unlabelled/train-0.parquet",
    ]
    assert plan.units(paths) == ["deu_Latn", "fra_Latn", "unlabelled"]


def test_pattern_of_a_unit():
    assert plan.pattern("fra_Latn") == "data/*/fra_Latn/*.parquet"
    assert plan.pattern("unlabelled") == "unlabelled/*.parquet"


def test_rows_select_the_routing_columns_and_add_a_missing_status(tmp_path):
    labelled = plan.rows(_shard(tmp_path / "a.parquet", [("1", "t", "fra_Latn", "wikipedia")]))
    assert labelled.columns == list(route.COLUMNS)
    assert labelled["status"].to_list() == [None]
    unlabelled = plan.rows(_shard(tmp_path / "b.parquet", [("1", "t", "fra_Latn", "website")], "unsplit"))
    assert unlabelled["status"].to_list() == ["unsplit"]


def test_detect_looks_at_website_rows_only():
    frame = pl.DataFrame(
        [("1", "Home", "hin_Deva", "website", None), ("2", "Home", "hin_Deva", "wikipedia", None)],
        schema={**SCHEMA, "status": pl.String},
        orient="row",
    )
    assert plan.detect(frame, _identify).to_dicts() == [{"id": "1", "detected": "eng_Latn"}]


def test_shards_cut_by_characters_in_order():
    work = pl.DataFrame(
        {"key": [b"a", b"b", b"c", b"d"], "translation_language": ["x"] * 4, "text": ["ab", "cd", "e", "fgh"]}
    )
    parts = plan.shards(work, chars=4)
    assert [part["text"].to_list() for part in parts] == [["ab", "cd"], ["e", "fgh"]]
    assert parts[0].columns == ["key", "translation_language", "text"]
    assert [p["text"].to_list() for p in plan.shards(work, chars=1)] == [["ab"], ["cd"], ["e"], ["fgh"]]


def test_a_unit_plan_routes_deduplicates_sorts_and_writes_its_files(tmp_path):
    unit = plan.Plan()
    rows = [
        ("1", "Bonjour tout le monde", "fra_Latn", "wikipedia"),
        ("2", "Salut", "fra_Latn", "wikipedia"),
        ("3", "Home", "fra_Latn", "website"),
        ("4", "Hallo Welt", "fra_Latn", "website"),
    ]
    unit.add(plan.rows(_shard(tmp_path / "a.parquet", rows)), _identify)
    unit.add(
        plan.rows(_shard(tmp_path / "b.parquet", [("5", "Salut", "fra_Latn", "osm_description")])), _identify
    )
    files = plan.write("fra_Latn", unit, tmp_path / "out")
    assert [remote for _, remote in files] == [
        "work/fra_Latn/00000.parquet",
        "routes/fra_Latn.parquet",
        "plan/fra_Latn.json",
    ]
    work = pl.read_parquet(files[0][0])
    assert work.select("translation_language", "text").rows() == [
        ("deu_Latn", "Hallo Welt"),
        ("fra_Latn", "Salut"),
        ("fra_Latn", "Bonjour tout le monde"),
    ]
    assert work["key"].to_list()[1] == route.key("fra_Latn", "Salut")
    assert pl.read_parquet(files[1][0]).to_dicts() == [
        {"id": "3", "detected": "eng_Latn"},
        {"id": "4", "detected": "deu_Latn"},
    ]
    assert json.loads(files[2][0].read_text()) == {"nllb": 4, "original": 1, "distinct": 3, "characters": 36}


def test_an_empty_unit_writes_no_work_and_empty_routes(tmp_path):
    files = plan.write("lat_Latn", plan.Plan(), tmp_path)
    assert [remote for _, remote in files] == ["routes/lat_Latn.parquet", "plan/lat_Latn.json"]
    assert pl.read_parquet(files[0][0]).schema == {"id": pl.String, "detected": pl.String}
    assert json.loads(files[1][0].read_text()) == {"distinct": 0, "characters": 0}
