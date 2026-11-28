import polars as pl

from georeset.label import backfill
from georeset.label.text import sha256_text


def corpus():
    texts = ["Forest.", "Born in 1900.", "Lake.", "Town."]
    return pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "text_sha256": [sha256_text(text) for text in texts],
            "landuse": [True, False, True, False],
            "group": ["website"] * 4,
        }
    )


def generations():
    website = {
        sha256_text("Forest."): "</think>yes",
        sha256_text("Born in 1900."): "</think>no",
        sha256_text("Lake."): "</think>no",
    }
    rows = [("website", text, raw) for text, raw in website.items()]
    rows.append(("description", sha256_text("Lake."), "</think>yes"))  # same text, another repository
    return pl.DataFrame(
        {
            "group": [row[0] for row in rows],
            "text_sha256": [row[1] for row in rows],
            "raw_output": [row[2] for row in rows],
            "finish_reason": ["stop"] * len(rows),
            "truncated": [False] * len(rows),
        }
    )


def test_round_rows_keep_each_row_id_once_with_its_label_and_failure():
    out = backfill.round_rows(corpus(), generations()).sort("id")
    assert out.columns == ["id", "text_sha256", "decision", "failure"]
    assert out["id"].to_list() == ["a", "b", "c", "d"]
    assert out["decision"].to_list() == ["yes", "no", "yes", "no"]
    assert out["failure"].to_list() == [None, None, None, backfill.MISSING]


def test_raw_rows_hold_each_group_and_text_once_with_its_output():
    out = backfill.raw_rows(corpus(), generations()).sort("text_sha256")
    assert out.columns == ["group", "text_sha256", "raw_output", "finish_reason", "truncated"]
    assert out.height == 3
    assert set(out["group"].to_list()) == {"website"}


def test_agreement_counts_where_the_raw_output_gives_back_the_label():
    stats = backfill.agreement(corpus(), generations())
    assert stats == {"rows": 4, "parsed": 3, "agree": 2, "disagree": 1, "missing": 1}
