import polars as pl
from conftest import build_all

from georeset.build.pipeline import build_meta, build_rows, region_chunks


def test_end_to_end(layout):
    build_all(layout)
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
    build_all(layout)
    first = pl.read_parquet(layout.out / "data/wikipedia")
    build_all(layout)
    assert pl.read_parquet(layout.out / "data/wikipedia").equals(first)
    assert list(layout.stage.glob("*/*")) == []
    assert list((layout.work / "rows").glob("data/*/*")) == []
    assert not (layout.work / "rows" / "unlabelled").exists()


def test_rows_skip_region_without_docs(layout):
    build_meta(layout)
    (layout.work / "meta" / "docs" / "italy-latest.parquet").unlink()
    build_rows(layout, layout.stage / "website" / "italy-latest")
    assert not (layout.work / "rows").exists()


def test_region_chunks_accumulate_until_the_budget():
    counts = pl.DataFrame({"region": ["a", "b", "c"], "n": [2, 2, 2]})
    assert region_chunks(counts, 5) == [["a", "b"], ["c"]]


def test_region_chunks_groups_consecutive_regions_within_budget():
    counts = pl.DataFrame({"region": ["c", "a", "b", "d"], "n": [5, 2, 2, 1]})
    assert region_chunks(counts, 4) == [["a", "b"], ["c"], ["d"]]
    assert region_chunks(counts, 100) == [["a", "b", "c", "d"]]
    assert region_chunks(counts.head(0), 4) == []


def test_large_partition_is_sorted_across_chunks(layout, monkeypatch):
    from georeset.build import pipeline

    monkeypatch.setattr(pipeline, "SORT_CHUNK_ROWS", 1)
    monkeypatch.setattr(pipeline, "ROWS_PER_SHARD", 1)
    build_all(layout)
    shards = sorted((layout.out / "data/wikipedia/fra_Latn").iterdir())
    assert len(shards) == 2
    assert pl.read_parquet(layout.out / "polygons")["polygon_id"].to_list() == ["way/1", "way/2", "way/3"]


def test_shards_are_sorted_by_region_then_document_then_position(layout, monkeypatch):
    from georeset.build import pipeline

    monkeypatch.setattr(pipeline, "SORT_CHUNK_ROWS", 1)
    rows = layout.work / "rows" / "data" / "s" / "l"
    rows.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "region": ["b", "a", "b", "a", "a"],
            "doc_key": ["d1", "d2", "d1", "d1", "d2"],
            "position": [1, 0, 0, 0, 1],
            "text": ["b-d1-1", "a-d2-0", "b-d1-0", "a-d1-0", "a-d2-1"],
        }
    )
    frame.head(2).write_parquet(rows / "x.parquet")
    frame.tail(3).write_parquet(rows / "y.parquet")
    pipeline._shard(rows, layout.out / "data" / "s" / "l")
    out = pl.read_parquet(layout.out / "data" / "s" / "l")
    assert out.columns == ["region", "text"]
    assert out["text"].to_list() == ["a-d1-0", "a-d2-0", "a-d2-1", "b-d1-0", "b-d1-1"]
    assert not rows.exists()


def test_rows_leave_a_done_marker_and_polygons_are_geoparquet_sorted(layout, monkeypatch):
    import json

    import pyarrow.parquet as pq

    from georeset.build import pipeline

    monkeypatch.setattr(pipeline, "POLYGONS_PER_SHARD", 2)
    build_all(layout)
    assert sorted(p.name for p in (layout.work / "rows" / "_done").iterdir()) == [
        "website__italy-latest",
        "wiki__centro-latest",
        "wiki__italy-latest",
    ]
    shards = sorted((layout.out / "polygons").iterdir())
    assert [s.name for s in shards] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert json.loads(pq.read_schema(shards[0]).metadata[b"geo"])["primary_column"] == "geometry"
    polygons = pl.read_parquet(layout.out / "polygons")
    assert polygons.select("region", "polygon_id").rows() == [
        ("centro", "way/1"),
        ("italy", "way/2"),
        ("italy", "way/3"),
    ]
