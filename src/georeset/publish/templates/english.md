<!-- files -->

embeddings_en/train-*.parquet              English translations of the relevant sentences, one vector each (see [Methodology](#methodology))
<!-- configs -->
, `embeddings_en`
<!-- method -->

$step. **Embeddings of the English translations.** The $embedded distinct `text_en` of the relevant sentences that NLLB-200 translated (`translation` = `nllb`; the smallest `id` keeps a repeated translation) embedded like `embeddings`, in the same space: [`$model`](https://huggingface.co/$model) (sentence-transformers, bf16, or float32 on GPUs without bf16; instruction `$prompt`, last-token pooling, at most $tokens tokens, L2-normalised), stored as float16 in `embeddings_en` with their `id`, `lat` and `lon`; join on `id` for the rest. Relevant sentences already in English (`translation` = `original`) have theirs in `embeddings`. On [Grid'5000](https://www.grid5000.fr) GPUs. Code: $code_url
