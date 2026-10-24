import polars as pl

from georeset.sources.base import SENTENCE_SCHEMA, attach_text, finish_sentences


def test_attach_text_picks_sentence_or_full_text():
    docs = pl.DataFrame(
        {
            "k": ["a", "b"],
            "sentences": [["s0", "s1"], None],
            "full_text": [None, "whole"],
        }
    )
    labels = pl.DataFrame(
        {
            "k": ["a", "a", "b"],
            "sentence_index": [1, 0, 0],
            "label": ["yes", "no", "skipped_unsplit"],
        },
        schema_overrides={"sentence_index": pl.Int32},
    )
    out = attach_text(labels, docs, ["k"])
    assert out["text"].to_list() == ["s1", "s0", "whole"]
    assert out.columns == ["k", "sentence_index", "label", "text"]


def test_attach_text_out_of_range_index_is_null():
    docs = pl.DataFrame({"k": ["a"], "sentences": [["s0"]], "full_text": [None]})
    labels = pl.DataFrame(
        {"k": ["a"], "sentence_index": [5], "label": ["yes"]},
        schema_overrides={"sentence_index": pl.Int32},
    )
    assert attach_text(labels, docs, ["k"])["text"].to_list() == [None]


def test_finish_sentences_filters_debris_and_normalizes():
    raw = pl.DataFrame(
        {
            "source": ["s", "s", "s"],
            "doc_key": ["d", "d", "d"],
            "position": [0, 1, 2],
            "text": ["Bonjour", "-", None],
            "label": ["yes", "no", "no"],
            "failure_reason": [None, None, None],
            "language": ["fr", "fr", "fr"],
            "url": ["u", "u", "u"],
            "extra": [1, 2, 3],
        }
    )
    out = finish_sentences(raw, "wikimedia")
    assert out.schema == SENTENCE_SCHEMA
    assert out["text"].to_list() == ["Bonjour"]
    assert out["language"].to_list() == ["fra_Latn"]
