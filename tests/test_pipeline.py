import polars as pl

from georeset.pipeline import build_meta, build_rows, build_shards


def build(layout):
    build_meta(layout)
    for staged in sorted(layout.stage.glob("*/*")):
        build_rows(layout, staged)
    build_shards(layout, workers=2)


def test_end_to_end(layout):
    build(layout)
    out = layout.out
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*.parquet")) == [
        "data/website/eng_Latn/train-00000-of-00001.parquet",
        "data/wikipedia/fra_Latn/train-00000-of-00001.parquet",
        "polygons/train-00000-of-00001.parquet",
        "unlabelled/train-00000-of-00001.parquet",
    ]
    wiki = pl.read_parquet(out / "data/wikipedia")
    assert wiki["landuse"].to_list() == [True, False]
    assert wiki["region"].to_list() == ["centro", "centro"]
    assert wiki["polygon_ids"].to_list() == [["way/1"], ["way/1"]]
    assert pl.read_parquet(out / "unlabelled").height == 1
    polygons = pl.read_parquet(out / "polygons").sort("polygon_id")
    assert polygons["region"].to_list() == ["centro", "italy", "italy"]
    assert polygons["name"].to_list() == [
        "centro-latest",
        "italy-latest",
        "italy-latest",
    ]


def test_rerun_is_a_no_op_and_work_is_cleaned(layout):
    build(layout)
    first = pl.read_parquet(layout.out / "data/wikipedia")
    build(layout)
    assert pl.read_parquet(layout.out / "data/wikipedia").equals(first)
    assert list(layout.stage.glob("*/*")) == []
    assert list((layout.work / "rows").glob("data/*/*")) == []
    assert not (layout.work / "rows" / "unlabelled").exists()


def test_rows_skip_region_without_docs(layout):
    build_meta(layout)
    (layout.work / "meta" / "docs" / "italy-latest.parquet").unlink()
    build_rows(layout, layout.stage / "website" / "italy-latest")
    assert not (layout.work / "rows").exists()
