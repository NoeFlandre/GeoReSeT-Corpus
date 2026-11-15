<!-- headline -->
 Every sentence also has an English version: $translated sentences machine-translated with NLLB-200, the rest already in English.
<!-- quickstart -->

English only, whatever the original language:

```python
english = load_dataset("$repo_id", split="train").select_columns(["id", "text_en", "translation", "landuse"])
```

<!-- fields -->

| `text_en` | string | The sentence in English: `text` itself when already English, else its NLLB translation; null when not translated (see [English translations](#english-translations)) |
| `translation` | string | Where `text_en` comes from: `original`, `nllb`, `unsupported` (language NLLB does not know) or `unsplit` (whole unsplit text, not translated) |
| `translation_language` | string | NLLB source code the sentence was translated from (`fra_Latn`, `zho_Hant`); null unless `translation = nllb` |
| `translation_flags` | list[string] | Failed checks on the translation (empty when none): `empty`, `repetition_loop`, `length_ratio_outlier`, `untranslated_copy`, `wrong_language`, `truncated`, `input_truncated` |
<!-- section -->

## English translations

$translated sentences were translated into English with [`facebook/nllb-200-3.3B`]($model_url) ($distinct distinct texts from $languages source languages, each translated once); the others are English already or were left out:

$kinds

The model was chosen on [`llms-xx-to-en-translation`](https://huggingface.co/datasets/NoeFlandre/llms-xx-to-en-translation), a benchmark of 17 open models on the corpus' languages into English, where NLLB-200 3.3B reaches chrF++ 66.4 and COMET-22 0.889; the settings below are the benchmark's.

* **Source language.** The row's `language`, except for `website` sentences, which carry their page's language: when GlotLID v4 gives a sentence another language with probability ≥ $threshold, that language is used (an English menu line on a Hindi page is kept as it is). Chinese (`cmn_Hani`) is read as Traditional (`zho_Hant`) when it has more Traditional-only than Simplified-only characters, else Simplified; `fas_Arab`, `ekk_Latn` and `yue_Hani` are NLLB's `pes_Arab`, `est_Latn` and `yue_Hant`; Serbian in Latin script is transliterated to Cyrillic first. Latin, Western Frisian, Uzbek in Cyrillic and romanized Indic and Arabic text are not supported (`unsupported`).
* **Decoding.** Greedy, float16, with [CTranslate2](https://github.com/OpenNMT/CTranslate2); each sentence alone (no document context); output budget 3× the source tokens, between 32 and 2,048.
* **Checks.** `translation_flags` lists the benchmark's failure checks (empty output, repeated n-grams, output length outside 0.33–3× the source's in words, a copy of the source, an output GlotLID v4 does not label English), plus `truncated` (the output budget was reached) and `input_truncated` (source over 1,022 tokens, cut):

$flags

<!-- files -->


Every row of `data/` and `unlabelled/` has the four translation columns (see [English translations](#english-translations)).
<!-- method -->

7. **English translations.** NLLB-200 3.3B ([revision `$model_short`]($model_tree)) through CTranslate2 on [Grid'5000](https://www.grid5000.fr) GPUs, with the settings of [`llms-xx-to-en-translation`](https://github.com/NoeFlandre/llms-xx-to-en-translation) (see [English translations](#english-translations)). Code: $code_url
<!-- license -->

* `text_en` of a `nllb` row is a machine translation of `text` and keeps `text`'s license. NLLB-200's weights are CC-BY-NC 4.0; the translations were produced for research.
