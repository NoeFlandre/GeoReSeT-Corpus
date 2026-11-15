"""The two models: NLLB-200 3.3B on CTranslate2 (GPU) and GlotLID v4 (CPU).

Imports of ``ctranslate2``, ``transformers`` and ``fasttext`` are deferred: they are installed only in the
Grid'5000 job environments (the ``translate`` extra).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xx_to_en.metric_runtime import GLOTLID_FILE, GLOTLID_REPO
from xx_to_en.scoring import glotlid_code

from georeset.translate.worker import MAX_INPUT_TOKENS, Output

NLLB_REPO = "facebook/nllb-200-3.3B"
NLLB_REVISION = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"  # as in the benchmark's runs
TARGET = "eng_Latn"
BATCH_TOKENS = 8192  # CTranslate2 sub-batches of at most this many source tokens


class Nllb:
    """Greedy NLLB translation into English, in float16 like the benchmark."""

    def __init__(
        self, model: Path, translator: Any = None, tokenizer: Any = None, batch_tokens: int = BATCH_TOKENS
    ):
        if translator is None:  # pragma: no cover (needs a GPU)
            import ctranslate2  # ty: ignore[unresolved-import]

            translator = ctranslate2.Translator(str(model), device="cuda", compute_type="float16")
        if tokenizer is None:  # pragma: no cover (needs the model files)
            from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

            tokenizer = AutoTokenizer.from_pretrained(model)
        self.translator, self.tokenizer, self.batch_tokens = translator, tokenizer, batch_tokens

    def _ids(self, texts: Sequence[str], language: str) -> list[list[int]]:
        """``[language] tokens </s>`` per text, as the benchmark's tokenizer calls produce."""
        self.tokenizer.src_lang = language
        return self.tokenizer(list(texts)).input_ids

    def count(self, texts: Sequence[str], language: str) -> list[int]:
        return [len(ids) - 2 for ids in self._ids(texts, language)]

    def translate(self, texts: Sequence[str], language: str, budgets: Sequence[int]) -> list[Output]:
        sources = [self.tokenizer.convert_ids_to_tokens(ids) for ids in self._ids(texts, language)]
        results = self.translator.translate_batch(
            sources,
            target_prefix=[[TARGET]] * len(sources),
            beam_size=1,
            max_decoding_length=max(budgets),
            max_input_length=MAX_INPUT_TOKENS,
            max_batch_size=self.batch_tokens,
            batch_type="tokens",
        )
        return [self._output(r.hypotheses[0], b) for r, b in zip(results, budgets, strict=True)]

    def _output(self, tokens: list[str], budget: int) -> Output:
        """The hypothesis (target token first) cut to its own budget, decoded."""
        kept = tokens[:budget]
        ids = self.tokenizer.convert_tokens_to_ids(kept)
        return Output(self.tokenizer.decode(ids, skip_special_tokens=True).strip(), len(kept))


class GlotLid:
    """GlotLID v4, the benchmark's language identifier; texts are flattened to one line."""

    def __init__(self, model: Any = None):
        if model is None:  # pragma: no cover (downloads the model)
            import fasttext  # ty: ignore[unresolved-import]
            from huggingface_hub import hf_hub_download

            model = fasttext.load_model(hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILE))
        self.model = model

    def predict(self, texts: Sequence[str]) -> tuple[list[str], list[float]]:
        """Label and probability per text; an empty text gets ``""`` and 0."""
        flat = [text.replace("\n", " ").strip() for text in texts]
        if not flat:
            return [], []
        labels, probabilities = self.model.predict(flat, k=1)
        pairs = list(map(_best, flat, labels, probabilities))
        return [label for label, _ in pairs], [p for _, p in pairs]

    def labels(self, texts: Sequence[str]) -> list[str]:
        return self.predict(texts)[0]


def _best(text: str, labels: Any, probabilities: Any) -> tuple[str, float]:
    return (glotlid_code(labels[0]), float(probabilities[0])) if text else ("", 0.0)
