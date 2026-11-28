import polars as pl

from georeset.label import partition
from georeset.label.text import sha256_text


def test_write_buckets_splits_rows_by_the_first_hex_digit(tmp_path):
    frame = pl.DataFrame({"text_sha256": ["0abc", "0def", "f123"], "v": [1, 2, 3]})
    partition.write_buckets(frame, tmp_path, "x")
    assert sorted(path.parent.name for path in tmp_path.glob("*/x.parquet")) == ["0", "f"]
    assert pl.read_parquet(tmp_path / "0" / "x.parquet")["v"].to_list() == [1, 2]


def test_corpus_file_rows_hash_each_text_and_map_each_source_to_its_group(tmp_path):
    path = tmp_path / "c.parquet"
    pl.DataFrame(
        {
            "id": ["a", "b"],
            "text": ["Forest.", "Lake."],
            "landuse": [True, False],
            "source": ["wikivoyage", "website"],
            "url": ["u", "v"],
        }
    ).write_parquet(path)
    out = partition.corpus_file_rows(path)
    assert out.columns == ["id", "text_sha256", "landuse", "group"]
    assert out["text_sha256"].to_list() == [sha256_text("Forest."), sha256_text("Lake.")]
    assert out["group"].to_list() == ["wikipedia", "website"]


def test_generation_file_rows_keep_the_round_columns_and_name_the_group(tmp_path):
    path = tmp_path / "g.parquet"
    pl.DataFrame(
        {
            "text_sha256": ["0abc"],
            "raw_output": ["</think>yes"],
            "finish_reason": ["stop"],
            "truncated": [False],
            "code_commit": ["c0ffee"],
        }
    ).write_parquet(path)
    out = partition.generation_file_rows(path, "website")
    assert out.columns == ["text_sha256", "raw_output", "finish_reason", "truncated", "group"]
    assert out["group"].to_list() == ["website"]


def test_join_bucket_matches_rows_within_their_own_group(tmp_path):
    sha = sha256_text("Forest.")
    corpus_dir, generations_dir = tmp_path / "corpus", tmp_path / "gen"
    partition.write_buckets(
        pl.DataFrame({"id": ["a"], "text_sha256": [sha], "landuse": [True], "group": ["website"]}),
        corpus_dir,
        "c",
    )
    partition.write_buckets(
        pl.DataFrame(
            {
                "group": ["website", "description"],
                "text_sha256": [sha, sha],
                "raw_output": ["</think>yes", "</think>no"],
                "finish_reason": ["stop", "stop"],
                "truncated": [False, False],
            }
        ),
        generations_dir,
        "g",
    )
    rounds, raws, stats = partition.join_bucket(sha[0], corpus_dir, generations_dir)
    assert rounds["decision"].to_list() == ["yes"]
    assert raws["group"].to_list() == ["website"]
    assert stats == {"rows": 1, "parsed": 1, "agree": 1, "disagree": 0, "missing": 0}


def test_join_bucket_with_nothing_in_it_is_empty(tmp_path):
    rounds, raws, stats = partition.join_bucket("0", tmp_path / "corpus", tmp_path / "gen")
    assert rounds.height == 0 and raws.height == 0
    assert stats["rows"] == 0
    assert rounds.columns == ["id", "text_sha256", "decision", "failure"]
