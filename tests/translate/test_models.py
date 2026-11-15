from types import SimpleNamespace

import numpy as np

from georeset.translate import models
from georeset.translate.worker import Output


class FakeTokenizer:
    """Tokens are words; ids are word lengths; the language token and ``</s>`` wrap the text."""

    def __init__(self):
        self.src_lang = None
        self.languages = []

    def __call__(self, texts):
        self.languages.append(self.src_lang)
        return SimpleNamespace(input_ids=[[0, *(len(w) for w in t.split()), 2] for t in texts])

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return " " + " ".join(i for i in ids if i != models.TARGET) + " "


class FakeTranslator:
    def __init__(self):
        self.kwargs = None

    def translate_batch(self, sources, **kwargs):
        self.kwargs = {"sources": sources, **kwargs}
        return [SimpleNamespace(hypotheses=[[models.TARGET, "a", "b", "c"]]) for _ in sources]


def _nllb():
    return models.Nllb(None, FakeTranslator(), FakeTokenizer(), batch_tokens=64)


def test_count_is_tokens_without_specials_in_the_source_language():
    nllb = _nllb()
    assert nllb.count(["un deux", "trois"], "fra_Latn") == [2, 1]
    assert nllb.tokenizer.languages == ["fra_Latn"]


def test_translate_decodes_greedily_into_english_and_cuts_each_output_to_its_budget():
    nllb = _nllb()
    outputs = nllb.translate(["un deux", "trois"], "fra_Latn", [3, 32])
    assert outputs == [Output("a b", 3), Output("a b c", 4)]
    assert nllb.translator.kwargs == {
        "sources": [["t0", "t2", "t4", "t2"], ["t0", "t5", "t2"]],
        "target_prefix": [[models.TARGET], [models.TARGET]],
        "beam_size": 1,
        "max_decoding_length": 32,
        "max_input_length": 1024,
        "max_batch_size": 64,
        "batch_type": "tokens",
    }


def test_default_batch_size():
    assert models.Nllb(None, FakeTranslator(), FakeTokenizer()).batch_tokens == models.BATCH_TOKENS


class FakeFastText:
    def __init__(self):
        self.seen = None

    def predict(self, texts, k):
        assert k == 1
        self.seen = texts
        return [["__label__eng_Latn"] for _ in texts], [np.array([0.5 + i / 10]) for i, _ in enumerate(texts)]


def test_glotlid_flattens_texts_and_strips_labels():
    lid = models.GlotLid(FakeFastText())
    assert lid.predict(["a\nb", "  ", "c"]) == (["eng_Latn", "", "eng_Latn"], [0.5, 0.0, 0.7])
    assert lid.model.seen == ["a b", "", "c"]
    assert lid.labels(["x"]) == ["eng_Latn"]
    assert lid.predict([]) == ([], [])
