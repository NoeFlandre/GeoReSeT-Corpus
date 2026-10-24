import json

import polars as pl
import pyarrow.parquet as pq

from georeset.build.write import write_shards


def test_shards_names_counts_and_order(tmp_path):
    df = pl.DataFrame({"x": list(range(10))})
    out = tmp_path / "data" / "wikipedia"
    paths = write_shards(df, out, rows_per_shard=4, row_group_size=2)
    assert [p.name for p in paths] == [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ]
    assert pl.read_parquet(out).equals(df)
    meta = pq.ParquetFile(paths[0]).metadata
    assert meta.num_row_groups == 2
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert meta.row_group(0).column(0).statistics.has_min_max


def test_rewrite_replaces_previous_shards(tmp_path):
    out = tmp_path / "d"
    write_shards(pl.DataFrame({"x": list(range(10))}), out, rows_per_shard=4)
    write_shards(pl.DataFrame({"x": [1]}), out, rows_per_shard=4)
    assert [p.name for p in out.iterdir()] == ["train-00000-of-00001.parquet"]


def test_empty_frame_writes_nothing(tmp_path):
    assert write_shards(pl.DataFrame({"x": []}), tmp_path / "d", rows_per_shard=4) == []
    assert not (tmp_path / "d").exists()


def test_geoparquet_metadata(tmp_path):
    df = pl.DataFrame({"geometry": [b"\x01"]})
    [path] = write_shards(df, tmp_path / "p", rows_per_shard=4, geometry="geometry")
    geo = json.loads(pq.read_schema(path).metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    assert geo["version"] == "1.1.0"


def test_no_geo_metadata_by_default(tmp_path):
    [path] = write_shards(pl.DataFrame({"x": [1]}), tmp_path / "p", rows_per_shard=4)
    assert b"geo" not in (pq.read_schema(path).metadata or {})


def test_stream_fills_shards_across_frame_boundaries(tmp_path):
    from georeset.build.write import write_stream

    frames = [pl.DataFrame({"x": [0, 1, 2]}), pl.DataFrame({"x": [3]}), pl.DataFrame({"x": [4, 5, 6, 7, 8]})]
    paths = write_stream(frames, 9, tmp_path / "d", rows_per_shard=4, row_group_size=2)
    assert [pl.read_parquet(p)["x"].to_list() for p in paths] == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert paths[-1].name == "train-00002-of-00003.parquet"


def test_stream_keeps_geo_metadata_on_every_shard(tmp_path):
    from georeset.build.write import write_stream

    frames = [pl.DataFrame({"geometry": [b"\x01", b"\x02", b"\x03"]})]
    paths = write_stream(frames, 3, tmp_path / "p", rows_per_shard=2, geometry="geometry")
    assert all(b"geo" in pq.read_schema(p).metadata for p in paths)


def test_stream_tops_up_a_partly_filled_shard_from_a_larger_frame(tmp_path):
    from georeset.build.write import write_stream

    frames = [pl.DataFrame({"x": [0, 1, 2]}), pl.DataFrame({"x": [3, 4, 5, 6, 7]})]
    paths = write_stream(frames, 8, tmp_path / "d", rows_per_shard=4)
    assert [pl.read_parquet(p)["x"].to_list() for p in paths] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_geo_metadata_is_exactly_geoparquet_1_1(tmp_path):
    [path] = write_shards(pl.DataFrame({"g": [b"\x01"]}), tmp_path / "p", rows_per_shard=4, geometry="g")
    assert json.loads(pq.read_schema(path).metadata[b"geo"]) == {
        "version": "1.1.0",
        "primary_column": "g",
        "columns": {"g": {"encoding": "WKB", "geometry_types": []}},
    }
