import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_translate_tokenizes_in_the_source_language():
    nllb = _nllb()
    nllb.translate(["un"], "fra_Latn", [8])
    assert nllb.tokenizer.languages == ["fra_Latn"]


def test_translate_refuses_a_result_count_mismatch():
    nllb = _nllb()
    nllb.translator.translate_batch = lambda sources, **kwargs: []
    with pytest.raises(ValueError, match="zip"):
        nllb.translate(["un"], "fra_Latn", [8])


def test_nllb_loads_the_model_on_the_gpu_in_float16(monkeypatch, tmp_path):
    loaded = []
    ctranslate2 = SimpleNamespace(Translator=lambda *a, **k: loaded.append(("translator", a, k)) or "t")
    tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: loaded.append(("tokenizer", a, k)) or "k")
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    nllb = models.Nllb(tmp_path)
    assert (nllb.translator, nllb.tokenizer) == ("t", "k")
    assert loaded == [
        ("translator", (str(tmp_path),), {"device": "cuda", "compute_type": "float16"}),
        ("tokenizer", (tmp_path,), {}),
    ]


def test_glotlid_downloads_the_benchmark_model(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: calls.append(k) or "glotlid.bin")
    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(load_model=lambda path: ("model", path)))
    assert models.GlotLid().model == ("model", "glotlid.bin")
    assert calls == [{"repo_id": models.GLOTLID_REPO, "filename": models.GLOTLID_FILE}]
