import polars as pl
import pytest

from georeset.translate import worker
from georeset.translate.worker import Output


class FakeEngine:
    """Counts one token per word; translates by upper-casing; records each call's budgets."""

    def __init__(self):
        self.calls = []
        self.counted = []

    def count(self, texts, language):
        self.counted.append(language)
        return [len(t.split()) for t in texts]

    def translate(self, texts, language, budgets):
        self.calls.append((language, list(budgets)))
        return [Output(f"en {t}", min(len(t.split()) + 1, b)) for t, b in zip(texts, budgets, strict=True)]


def _english(texts):
    return ["eng_Latn"] * len(texts)


def _work(rows):
    return pl.DataFrame(
        rows, schema={"key": pl.Binary, "translation_language": pl.String, "text": pl.String}, orient="row"
    )


def test_flags_name_the_failed_checks():
    assert worker.flags("Bonjour le monde", Output("Hello world", 3), 32, 3, "eng_Latn") == []
    assert worker.flags("Bonjour", Output("", 1), 32, 1, "") == ["empty", "length_ratio_outlier"]
    assert worker.flags("Bonjour le monde", Output("Bonjour le monde", 4), 32, 3, "fra_Latn") == [
        "untranslated_copy",
        "wrong_language",
    ]
    assert worker.flags("a b c", Output("x x x x x x x x x x", 32), 32, 3, "eng_Latn") == [
        "repetition_loop",
        "length_ratio_outlier",
        "truncated",
    ]
    assert worker.flags("a", Output("b", 31), 32, 1, "eng_Latn") == []
    assert worker.flags("a", Output("b", 2), 32, worker.MAX_INPUT_TOKENS - 1, "eng_Latn") == [
        "input_truncated"
    ]
    assert worker.flags("a", Output("b", 2), 32, worker.MAX_INPUT_TOKENS - 2, "eng_Latn") == []


def test_translate_group_gives_each_chunk_its_own_budgets(monkeypatch):
    monkeypatch.setattr(worker, "CHUNK", 2)
    engine = FakeEngine()
    texts = ["a", "b c", " ".join(["w"] * 20), " ".join(["v"] * 340), " ".join(["u"] * 341)]
    outputs, budgets, counts = worker.translate_group(engine, texts, "fra_Latn")
    assert counts == [1, 2, 20, 340, 341]
    assert budgets == [32, 32, 60, 1020, worker.MAX_OUTPUT_TOKENS]
    assert worker.MAX_OUTPUT_TOKENS == 1020
    assert engine.calls == [("fra_Latn", [32, 32]), ("fra_Latn", [60, 1020]), ("fra_Latn", [1020])]
    assert engine.counted == ["fra_Latn"]
    assert [o.text for o in outputs][:3] == ["en a", "en b c", "en " + " ".join(["w"] * 20)]


def test_translate_shard_groups_by_language_and_keeps_keys():
    engine = FakeEngine()
    work = _work([(b"k1", "deu_Latn", "hallo welt"), (b"k2", "fra_Latn", "salut"), (b"k3", "deu_Latn", "ja")])
    out = worker.translate_shard(engine, _english, work)
    assert out.rows() == [(b"k1", "en hallo welt", []), (b"k3", "en ja", []), (b"k2", "en salut", [])]
    assert out.schema == {"key": pl.Binary, "text_en": pl.String, "translation_flags": pl.List(pl.String)}
    assert [language for language, _ in engine.calls] == ["deu_Latn", "fra_Latn"]


def test_translate_shard_refuses_a_language_label_per_output_mismatch():
    work = _work([(b"k1", "deu_Latn", "hallo"), (b"k2", "deu_Latn", "welt")])
    with pytest.raises(ValueError, match="zip"):
        worker.translate_shard(FakeEngine(), lambda texts: ["eng_Latn"], work)


def test_translate_shard_keeps_the_languages_in_their_first_seen_order():
    languages = [f"l{i:02d}_Latn" for i in reversed(range(40))]
    work = _work([(lang.encode(), lang, "x") for lang in languages])
    out = worker.translate_shard(FakeEngine(), _english, work)
    assert out["key"].to_list() == [lang.encode() for lang in languages]
