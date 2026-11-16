---
license: ["odbl", "cc-by-sa-4.0", "other"]
pretty_name: "GeoReSeT Corpus"
language: ["fra"]
task_categories: ["text-classification"]
tags: ["openstreetmap", "land-use", "land-cover", "geospatial", "multilingual"]
size_categories: ["n<1K"]
configs: [{"config_name": "all", "data_files": [{"split": "train", "path": "data/*/*/*.parquet"}], "default": true}, {"config_name": "wikipedia", "data_files": [{"split": "train", "path": "data/wikipedia/*/*.parquet"}]}, {"config_name": "fra_Latn", "data_files": [{"split": "train", "path": "data/*/fra_Latn/*.parquet"}]}, {"config_name": "unlabelled", "data_files": [{"split": "train", "path": "unlabelled/*.parquet"}]}, {"config_name": "polygons", "data_files": [{"split": "train", "path": "polygons/*.parquet"}]}, {"config_name": "embeddings", "data_files": [{"split": "train", "path": "embeddings/*.parquet"}]}, {"config_name": "polygons_labels_shares", "data_files": [{"split": "train", "path": "polygons_labels_shares/*.parquet"}]}]
---

![GeoReSeT: collect texts linked to OpenStreetMap polygons, filter them by land-use relevance, link sentences back to polygons](assets/hero.png)

# GeoReSeT Corpus

**10 sentences in 1 languages, tied to 7 OpenStreetMap polygons in 2 regions, each labelled for land-use / land-cover relevance by an LLM.** English text comes with them: 7 sentences machine-translated with NLLB-200 and 3 already in English (see [English translations](#english-translations)).

Sentences come from the websites, `description` tags and Wikipedia / Wikivoyage articles linked to OpenStreetMap polygons. Every row is one sentence with a boolean `landuse` label, the language, and the location of the polygon it describes. Polygon shapes are in a separate `polygons` table.

**The full dataset**, all sentences:

![The full dataset per H3 cell](assets/map.png)

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("me/data", "fra_Latn", split="train")  # one language
ds = load_dataset("me/data", "wikipedia", split="train")  # one source
ds = load_dataset("me/data", split="train")  # everything
afg = load_dataset(
    "me/data", "wikipedia", split="train", filters=[("region", "==", "afghanistan")]
)  # one region
```

Join sentences to polygon shapes with DuckDB, reading only what the query needs:

```sql
SELECT s.text, s.landuse, p.osm_tag, p.geometry
FROM 'hf://datasets/me/data/data/wikipedia/*/*.parquet' s
JOIN 'hf://datasets/me/data/polygons/*.parquet' p ON p.polygon_id = s.polygon_ids[1]
WHERE s.region = 'afghanistan' AND s.landuse;
```

English only, whatever the original language:

```python
english = load_dataset("me/data", split="train").select_columns(["id", "text_en", "translation", "landuse"])
```

## Sentences

**Sentences per source**: `website`, `osm_description`, then `wikipedia` and `wikivoyage` together:

![Website sentences per H3 cell](assets/map_website.png)

![OSM description sentences per H3 cell](assets/map_description.png)

![Wikipedia and Wikivoyage sentences per H3 cell](assets/map_wikimedia.png)

**Sentences the LLM labelled relevant** (`landuse` is true):

![Sentences labelled relevant per H3 cell](assets/map_landuse.png)

| Source | Sentences | Words | Polygons | Languages | Regions | `landuse` share |
|---|---:|---:|---:|---:|---:|---:|
| wikipedia | 10 | 50 | 3 | 1 | 2 | 40.0% |
| **total** | 10 | 50 | 3 | 1 | 2 | 40.0% |

`polygons` counts distinct polygons linked to the sentences; the `polygons` table holds all 7. Words are whitespace-separated tokens, so Chinese, Japanese and Thai are undercounted.

![Sentences per language, top 20](assets/languages.png)

| Continent | Sentences | Polygons |
|---|---:|---:|
| europe | 10 | 3 |

Polygon areas, over all 7 polygons:

| 10% | 25% | Median | 75% | 90% |
|---:|---:|---:|---:|---:|
| 10 m² | 25 m² | 500 m² | 2.0 ha | 3.0 km² |

### Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable id of the sentence (128-bit hex) |
| `text` | string | The sentence, byte-exact as labelled |
| `landuse` | bool | The sentence describes land use, land cover or a geographic environment observable by remote sensing (LLM label) |
| `language` | string | ISO 639-3 + script (`fra_Latn`), see [Languages](#languages) |
| `source` | string | `website`, `osm_description`, `wikipedia` or `wikivoyage` |
| `polygon_ids` | list[string] | All polygons the sentence's document is linked to (`way/123`, `relation/45`) |
| `region` | string | Geofabrik extract of the first polygon (`afghanistan`, `us-texas`) |
| `continent` | string | Continent of that region |
| `lat`, `lon` | float | Centroid of the first polygon (WGS 84) |
| `name` | string | OSM `name` of the first polygon |
| `osm_tag` | string | Primary OSM tag of the first polygon (`landuse=forest`) |
| `area_m2` | float | Area of the first polygon, m² |
| `url` | string | Website, OSM object, or Wikipedia / Wikivoyage revision permalink |
| `license` | string | License of the text: `ODbL-1.0`, `CC-BY-SA-4.0` or `other` |
| `text_en` | string | The sentence in English: `text` itself when already English, else its NLLB translation; null when not translated (see [English translations](#english-translations)) |
| `translation` | string | Where `text_en` comes from: `original`, `nllb`, `unsupported` (language NLLB does not know) or `unsplit` (whole unsplit text, not translated) |
| `translation_language` | string | NLLB source code the sentence was translated from (`fra_Latn`, `zho_Hant`); null unless `translation = nllb` |
| `translation_flags` | list[string] | Failed checks on the translation (empty when none): `empty`, `repetition_loop`, `length_ratio_outlier`, `untranslated_copy`, `wrong_language`, `truncated`, `input_truncated` |

The `polygons` config holds one row per polygon: `polygon_id`, `osm_type`, `osm_id`, `name`, `osm_tag`, `region`, `continent`, `lat`, `lon`, `area_m2` and `geometry` (WKB, [GeoParquet](https://geoparquet.org) 1.1, WGS 84).

### Languages

Labels follow GlotLID: ISO 639-3 + ISO 15924 script. How the language was found depends on the source:

* `website`: [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) on the whole page, so a sentence carries its page's label (an English menu line on a Hindi page is `hin_Deva`).
* `osm_description`: [Lingua](https://github.com/pemistahl/lingua-py), with GlotLID v3 as fallback.
* `wikipedia`, `wikivoyage`: the language edition of the article (`fr.wikipedia.org` → `fra_Latn`), not detected.

Only text in the 85 languages of the [SaT](https://huggingface.co/segment-any-text/sat-3l-sm) sentence splitter was split and labelled.
Labels are finer than that: individual languages of a macrolanguage (`nob_Latn` and `nno_Latn` for Norwegian, `cmn_Hani` and `yue_Hani` for Chinese)
and scripts (`srp_Cyrl` and `srp_Latn`) stay distinct. Other languages appear only in `unlabelled`, as `unsplit`.

### Unlabelled sentences

The `unlabelled` config holds sentences without a usable label, kept to be labelled later. It has the same fields, with `status` and `failure_reason` instead of `landuse`:

* `failed`: the model gave no usable yes/no answer (`failure_reason`, nearly always `truncated`).
* `unsplit`: text the upstream splitter could not segment (unsupported language); `text` is a whole page, description or paragraph.

| Source | `failed` | `unsplit` |
|---|---:|---:|
| wikipedia | 1 | 2 |
| **total** | 1 | 2 |

## English translations

7 sentences were translated into English with [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) (6 distinct texts from 2 source languages, each translated once); the others are English already or were left out:

| `translation` | Sentences |
|---:|---:|
| `nllb` | 7 |
| `original` | 3 |

The model was chosen on [`llms-xx-to-en-translation`](https://huggingface.co/datasets/NoeFlandre/llms-xx-to-en-translation), a benchmark of 17 open models on the corpus' languages into English, where NLLB-200 3.3B reaches chrF++ 66.4 and COMET-22 0.889; the settings below are the benchmark's.

* **Source language.** The row's `language`, except for `website` sentences, which carry their page's language: when GlotLID v4 gives a sentence another language with probability ≥ 0.9, that language is used (an English menu line on a Hindi page is kept as it is). Chinese (`cmn_Hani`) is read as Traditional (`zho_Hant`) when it has more Traditional-only than Simplified-only characters, else Simplified; `fas_Arab`, `ekk_Latn` and `yue_Hani` are NLLB's `pes_Arab`, `est_Latn` and `yue_Hant`; Serbian in Latin script is transliterated to Cyrillic first. Latin, Western Frisian, Uzbek in Cyrillic and romanized Indic and Arabic text are not supported (`unsupported`).
* **Decoding.** Greedy, float16, with [CTranslate2](https://github.com/OpenNMT/CTranslate2); each sentence alone (no document context); output budget 3× the source tokens, between 32 and 2,048 as in the benchmark, capped at 1,020 because CTranslate2 has no decoder position past 1,023 (outputs reaching the cap are flagged `truncated`).
* **Checks.** `translation_flags` lists the benchmark's failure checks (empty output, repeated n-grams, output length outside 0.33–3× the source's in words, a copy of the source, an output GlotLID v4 does not label English), plus `truncated` (the output budget was reached) and `input_truncated` (source over 1,022 tokens, cut):

| Check | Sentences | Share of translated |
|---|---:|---:|
| truncated | 1 | 14.3% |

## EUNIS labels

`polygons_labels_shares` gives each polygon the share of each [EUNIS](https://zenodo.org/records/16985100) habitat class over the pixels of the EUNIS wall-to-wall map whose centre lies inside it (`eunis_code`, `eunis_shares`, `max_eunis_share_code`, `max_eunis_share_label`), with the polygon's fields; join on `polygon_id`. If no pixel centre lies inside, the pixel under the polygon's centroid is used (`label_method` = `centroid`; otherwise `polygon`), with shares `[1.0]`. `unlabelled_reason` gives why a polygon has no label. Labels are null outside the map's coverage and on nodata pixels: 6 of the 7 polygons have one. Copied from [`me/fused` @ `fffffff`](https://huggingface.co/datasets/me/fused/tree/ffffffffffffffffffffffffffffffffffffffff).
## Files

```
data/<source>/<language>/train-*.parquet   labelled sentences, sorted by region
unlabelled/train-*.parquet                 sentences without a label (see above)
polygons/train-*.parquet                   polygon shapes (GeoParquet)
embeddings/train-*.parquet                 relevant sentences, one vector each (see [Methodology](#methodology))
polygons_labels_shares/train-*.parquet     one row per polygon, with its EUNIS label and class shares (see [EUNIS labels](#eunis-labels))
```

Configs: `all` (default), one per source, one per language, `unlabelled`, `polygons`, `embeddings`, `polygons_labels_shares`.

Every row of `data/` and `unlabelled/` has the four translation columns (see [English translations](#english-translations)).

## Methodology

1. **Polygons.** Closed ways and multipolygon relations from the 386 regional [Geofabrik](https://download.geofabrik.de) extracts of OpenStreetMap, assembled with [osmium](https://osmcode.org/libosmium/).
2. **Text.** Three kinds of text attached to a polygon:
   * websites in `website` / `contact:website` tags, fetched and extracted with [Trafilatura](https://trafilatura.readthedocs.io): [`osm-polygon-website-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag) ([code](https://github.com/NoeFlandre/osm-polygon-website-tag));
   * `description` and `description:*` tags: [`osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) ([code](https://github.com/NoeFlandre/osm-polygon-description-tag));
   * Wikipedia and Wikivoyage articles reached through the `wikidata` tag: [`osm-polygon-wikidata-and-wikipedia`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) ([code](https://github.com/NoeFlandre/osm-polygon-wikidata-only)).
3. **Language.** [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) for websites, [Lingua](https://github.com/pemistahl/lingua-py) with GlotLID fallback for descriptions, the language edition for Wikipedia and Wikivoyage (see [Languages](#languages)).
4. **Sentences.** Split with [SaT `sat-3l-sm`](https://huggingface.co/segment-any-text/sat-3l-sm) in its 85 languages; text in other languages is left whole (`unsplit`).
5. **Labels.** Each sentence is classified by [`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B): does it describe land use, land cover or a geographic environment observable by remote sensing? Greedy decoding with reasoning, up to 4,096 tokens, served with SGLang and the [DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) draft model, on about 4,500 GPU-hours of [Grid'5000](https://www.grid5000.fr). The model and prompt were chosen on [`benchmark-llms-landuse-relevance`](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance) (53 models, 25,500 labelled sentences in 85 languages), where this setup reaches macro F1 0.81 and balanced accuracy 0.77. Labels and raw model outputs, with reasoning ([code](https://github.com/NoeFlandre/filter-osm-datasets-llms-landuse)):
   * [`me/up` @ `abc1234`](https://huggingface.co/datasets/me/up/tree/abc1234def)
6. **This dataset.** One row per sentence of each document (a document linked to several polygons appears once); overlapping extracts resolved to the most specific one; language codes normalized; sentences with fewer than two letters dropped, and Wikipedia / Wikivoyage rows made only of bracketed edit links or citations (`[ edit | edit source ]`, `[Smith 1998]`) removed. Join the raw outputs on `sha256(text)` = `text_sha256`. Code: https://github.com/me/code

7. **English translations.** NLLB-200 3.3B ([revision `1a07f7d`](https://huggingface.co/facebook/nllb-200-3.3B/tree/1a07f7d195896b2114afcb79b7b57ab512e7b43e)) through CTranslate2 on [Grid'5000](https://www.grid5000.fr) GPUs, with the settings of [`llms-xx-to-en-translation`](https://github.com/NoeFlandre/llms-xx-to-en-translation) (see [English translations](#english-translations)). Code: https://github.com/me/code
8. **Embeddings of the relevant sentences.** The 5 distinct relevant sentences embedded with [`me/embedder`](https://huggingface.co/me/embedder) (sentence-transformers, bf16, instruction `Instruct: what\nQuery: `, last-token pooling, L2-normalised), stored as float16 in `embeddings` with their `id`, `lat` and `lon`; join on `id` for the rest. Copied from [`me/fused` @ `fffffff`](https://huggingface.co/datasets/me/fused/tree/ffffffffffffffffffffffffffffffffffffffff).
## License

* OpenStreetMap data (`osm_description` text, polygons, names, tags): [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors.
* Wikipedia and Wikivoyage text: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); `url` points to the exact revision for attribution.
* Website text (`license = other`): © the respective site owners, included for research. To have content removed, open a discussion on this repository.

* `text_en` of a `nllb` row is a machine translation of `text` and keeps `text`'s license. NLLB-200's weights are CC-BY-NC 4.0; the translations were produced for research.