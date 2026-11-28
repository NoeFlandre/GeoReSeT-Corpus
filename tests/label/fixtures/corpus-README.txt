---
license: ["odbl", "cc-by-sa-4.0", "other"]
pretty_name: "GeoReSeT Corpus"
language: ["afr", "als", "amh", "arb", "azj", "bel", "ben", "bul", "cat", "ceb", "ces", "ckb", "cmn", "cym", "dan", "deu", "ekk", "ell", "eng", "epo", "eus", "fas", "fin", "fra", "fry", "gla", "gle", "glg", "guj", "hau", "heb", "hin", "hun", "hye", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kat", "kaz", "khk", "khm", "kir", "kmr", "kor", "lat", "lit", "lvs", "mal", "mar", "mkd", "mlt", "mya", "nld", "nno", "nob", "npi", "pan", "pbt", "plt", "pol", "por", "ron", "rus", "sin", "slk", "slv", "spa", "srp", "swe", "tam", "tel", "tgk", "tha", "tur", "ukr", "urd", "uzn", "vie", "xho", "ydd", "yor", "yue", "zsm", "zul"]
task_categories: ["text-classification"]
tags: ["openstreetmap", "land-use", "land-cover", "geospatial", "multilingual"]
size_categories: ["10M<n<100M"]
configs: [{"config_name": "all", "data_files": [{"split": "train", "path": "data/*/*/*.parquet"}], "default": true}, {"config_name": "osm_description", "data_files": [{"split": "train", "path": "data/osm_description/*/*.parquet"}]}, {"config_name": "website", "data_files": [{"split": "train", "path": "data/website/*/*.parquet"}]}, {"config_name": "wikipedia", "data_files": [{"split": "train", "path": "data/wikipedia/*/*.parquet"}]}, {"config_name": "wikivoyage", "data_files": [{"split": "train", "path": "data/wikivoyage/*/*.parquet"}]}, {"config_name": "afr_Latn", "data_files": [{"split": "train", "path": "data/*/afr_Latn/*.parquet"}]}, {"config_name": "als_Latn", "data_files": [{"split": "train", "path": "data/*/als_Latn/*.parquet"}]}, {"config_name": "amh_Ethi", "data_files": [{"split": "train", "path": "data/*/amh_Ethi/*.parquet"}]}, {"config_name": "arb_Arab", "data_files": [{"split": "train", "path": "data/*/arb_Arab/*.parquet"}]}, {"config_name": "arb_Latn", "data_files": [{"split": "train", "path": "data/*/arb_Latn/*.parquet"}]}, {"config_name": "azj_Latn", "data_files": [{"split": "train", "path": "data/*/azj_Latn/*.parquet"}]}, {"config_name": "bel_Cyrl", "data_files": [{"split": "train", "path": "data/*/bel_Cyrl/*.parquet"}]}, {"config_name": "ben_Beng", "data_files": [{"split": "train", "path": "data/*/ben_Beng/*.parquet"}]}, {"config_name": "ben_Latn", "data_files": [{"split": "train", "path": "data/*/ben_Latn/*.parquet"}]}, {"config_name": "bul_Cyrl", "data_files": [{"split": "train", "path": "data/*/bul_Cyrl/*.parquet"}]}, {"config_name": "cat_Latn", "data_files": [{"split": "train", "path": "data/*/cat_Latn/*.parquet"}]}, {"config_name": "ceb_Latn", "data_files": [{"split": "train", "path": "data/*/ceb_Latn/*.parquet"}]}, {"config_name": "ces_Latn", "data_files": [{"split": "train", "path": "data/*/ces_Latn/*.parquet"}]}, {"config_name": "ckb_Arab", "data_files": [{"split": "train", "path": "data/*/ckb_Arab/*.parquet"}]}, {"config_name": "cmn_Hani", "data_files": [{"split": "train", "path": "data/*/cmn_Hani/*.parquet"}]}, {"config_name": "cym_Latn", "data_files": [{"split": "train", "path": "data/*/cym_Latn/*.parquet"}]}, {"config_name": "dan_Latn", "data_files": [{"split": "train", "path": "data/*/dan_Latn/*.parquet"}]}, {"config_name": "deu_Latn", "data_files": [{"split": "train", "path": "data/*/deu_Latn/*.parquet"}]}, {"config_name": "ekk_Latn", "data_files": [{"split": "train", "path": "data/*/ekk_Latn/*.parquet"}]}, {"config_name": "ell_Grek", "data_files": [{"split": "train", "path": "data/*/ell_Grek/*.parquet"}]}, {"config_name": "eng_Latn", "data_files": [{"split": "train", "path": "data/*/eng_Latn/*.parquet"}]}, {"config_name": "epo_Latn", "data_files": [{"split": "train", "path": "data/*/epo_Latn/*.parquet"}]}, {"config_name": "eus_Latn", "data_files": [{"split": "train", "path": "data/*/eus_Latn/*.parquet"}]}, {"config_name": "fas_Arab", "data_files": [{"split": "train", "path": "data/*/fas_Arab/*.parquet"}]}, {"config_name": "fin_Latn", "data_files": [{"split": "train", "path": "data/*/fin_Latn/*.parquet"}]}, {"config_name": "fra_Latn", "data_files": [{"split": "train", "path": "data/*/fra_Latn/*.parquet"}]}, {"config_name": "fry_Latn", "data_files": [{"split": "train", "path": "data/*/fry_Latn/*.parquet"}]}, {"config_name": "gla_Latn", "data_files": [{"split": "train", "path": "data/*/gla_Latn/*.parquet"}]}, {"config_name": "gle_Latn", "data_files": [{"split": "train", "path": "data/*/gle_Latn/*.parquet"}]}, {"config_name": "glg_Latn", "data_files": [{"split": "train", "path": "data/*/glg_Latn/*.parquet"}]}, {"config_name": "guj_Gujr", "data_files": [{"split": "train", "path": "data/*/guj_Gujr/*.parquet"}]}, {"config_name": "hau_Latn", "data_files": [{"split": "train", "path": "data/*/hau_Latn/*.parquet"}]}, {"config_name": "heb_Hebr", "data_files": [{"split": "train", "path": "data/*/heb_Hebr/*.parquet"}]}, {"config_name": "hin_Deva", "data_files": [{"split": "train", "path": "data/*/hin_Deva/*.parquet"}]}, {"config_name": "hun_Latn", "data_files": [{"split": "train", "path": "data/*/hun_Latn/*.parquet"}]}, {"config_name": "hye_Armn", "data_files": [{"split": "train", "path": "data/*/hye_Armn/*.parquet"}]}, {"config_name": "ibo_Latn", "data_files": [{"split": "train", "path": "data/*/ibo_Latn/*.parquet"}]}, {"config_name": "ind_Latn", "data_files": [{"split": "train", "path": "data/*/ind_Latn/*.parquet"}]}, {"config_name": "isl_Latn", "data_files": [{"split": "train", "path": "data/*/isl_Latn/*.parquet"}]}, {"config_name": "ita_Latn", "data_files": [{"split": "train", "path": "data/*/ita_Latn/*.parquet"}]}, {"config_name": "jav_Latn", "data_files": [{"split": "train", "path": "data/*/jav_Latn/*.parquet"}]}, {"config_name": "jpn_Jpan", "data_files": [{"split": "train", "path": "data/*/jpn_Jpan/*.parquet"}]}, {"config_name": "kan_Knda", "data_files": [{"split": "train", "path": "data/*/kan_Knda/*.parquet"}]}, {"config_name": "kan_Latn", "data_files": [{"split": "train", "path": "data/*/kan_Latn/*.parquet"}]}, {"config_name": "kat_Geor", "data_files": [{"split": "train", "path": "data/*/kat_Geor/*.parquet"}]}, {"config_name": "kaz_Cyrl", "data_files": [{"split": "train", "path": "data/*/kaz_Cyrl/*.parquet"}]}, {"config_name": "khk_Cyrl", "data_files": [{"split": "train", "path": "data/*/khk_Cyrl/*.parquet"}]}, {"config_name": "khm_Khmr", "data_files": [{"split": "train", "path": "data/*/khm_Khmr/*.parquet"}]}, {"config_name": "kir_Cyrl", "data_files": [{"split": "train", "path": "data/*/kir_Cyrl/*.parquet"}]}, {"config_name": "kmr_Latn", "data_files": [{"split": "train", "path": "data/*/kmr_Latn/*.parquet"}]}, {"config_name": "kor_Hang", "data_files": [{"split": "train", "path": "data/*/kor_Hang/*.parquet"}]}, {"config_name": "lat_Latn", "data_files": [{"split": "train", "path": "data/*/lat_Latn/*.parquet"}]}, {"config_name": "lit_Latn", "data_files": [{"split": "train", "path": "data/*/lit_Latn/*.parquet"}]}, {"config_name": "lvs_Latn", "data_files": [{"split": "train", "path": "data/*/lvs_Latn/*.parquet"}]}, {"config_name": "mal_Mlym", "data_files": [{"split": "train", "path": "data/*/mal_Mlym/*.parquet"}]}, {"config_name": "mar_Deva", "data_files": [{"split": "train", "path": "data/*/mar_Deva/*.parquet"}]}, {"config_name": "mar_Latn", "data_files": [{"split": "train", "path": "data/*/mar_Latn/*.parquet"}]}, {"config_name": "mkd_Cyrl", "data_files": [{"split": "train", "path": "data/*/mkd_Cyrl/*.parquet"}]}, {"config_name": "mlt_Latn", "data_files": [{"split": "train", "path": "data/*/mlt_Latn/*.parquet"}]}, {"config_name": "mya_Mymr", "data_files": [{"split": "train", "path": "data/*/mya_Mymr/*.parquet"}]}, {"config_name": "nld_Latn", "data_files": [{"split": "train", "path": "data/*/nld_Latn/*.parquet"}]}, {"config_name": "nno_Latn", "data_files": [{"split": "train", "path": "data/*/nno_Latn/*.parquet"}]}, {"config_name": "nob_Latn", "data_files": [{"split": "train", "path": "data/*/nob_Latn/*.parquet"}]}, {"config_name": "npi_Deva", "data_files": [{"split": "train", "path": "data/*/npi_Deva/*.parquet"}]}, {"config_name": "pan_Guru", "data_files": [{"split": "train", "path": "data/*/pan_Guru/*.parquet"}]}, {"config_name": "pbt_Arab", "data_files": [{"split": "train", "path": "data/*/pbt_Arab/*.parquet"}]}, {"config_name": "plt_Latn", "data_files": [{"split": "train", "path": "data/*/plt_Latn/*.parquet"}]}, {"config_name": "pol_Latn", "data_files": [{"split": "train", "path": "data/*/pol_Latn/*.parquet"}]}, {"config_name": "por_Latn", "data_files": [{"split": "train", "path": "data/*/por_Latn/*.parquet"}]}, {"config_name": "ron_Latn", "data_files": [{"split": "train", "path": "data/*/ron_Latn/*.parquet"}]}, {"config_name": "rus_Cyrl", "data_files": [{"split": "train", "path": "data/*/rus_Cyrl/*.parquet"}]}, {"config_name": "sin_Sinh", "data_files": [{"split": "train", "path": "data/*/sin_Sinh/*.parquet"}]}, {"config_name": "slk_Latn", "data_files": [{"split": "train", "path": "data/*/slk_Latn/*.parquet"}]}, {"config_name": "slv_Latn", "data_files": [{"split": "train", "path": "data/*/slv_Latn/*.parquet"}]}, {"config_name": "spa_Latn", "data_files": [{"split": "train", "path": "data/*/spa_Latn/*.parquet"}]}, {"config_name": "srp_Cyrl", "data_files": [{"split": "train", "path": "data/*/srp_Cyrl/*.parquet"}]}, {"config_name": "srp_Latn", "data_files": [{"split": "train", "path": "data/*/srp_Latn/*.parquet"}]}, {"config_name": "swe_Latn", "data_files": [{"split": "train", "path": "data/*/swe_Latn/*.parquet"}]}, {"config_name": "tam_Taml", "data_files": [{"split": "train", "path": "data/*/tam_Taml/*.parquet"}]}, {"config_name": "tel_Latn", "data_files": [{"split": "train", "path": "data/*/tel_Latn/*.parquet"}]}, {"config_name": "tel_Telu", "data_files": [{"split": "train", "path": "data/*/tel_Telu/*.parquet"}]}, {"config_name": "tgk_Cyrl", "data_files": [{"split": "train", "path": "data/*/tgk_Cyrl/*.parquet"}]}, {"config_name": "tha_Thai", "data_files": [{"split": "train", "path": "data/*/tha_Thai/*.parquet"}]}, {"config_name": "tur_Latn", "data_files": [{"split": "train", "path": "data/*/tur_Latn/*.parquet"}]}, {"config_name": "ukr_Cyrl", "data_files": [{"split": "train", "path": "data/*/ukr_Cyrl/*.parquet"}]}, {"config_name": "urd_Arab", "data_files": [{"split": "train", "path": "data/*/urd_Arab/*.parquet"}]}, {"config_name": "uzn_Cyrl", "data_files": [{"split": "train", "path": "data/*/uzn_Cyrl/*.parquet"}]}, {"config_name": "uzn_Latn", "data_files": [{"split": "train", "path": "data/*/uzn_Latn/*.parquet"}]}, {"config_name": "vie_Latn", "data_files": [{"split": "train", "path": "data/*/vie_Latn/*.parquet"}]}, {"config_name": "xho_Latn", "data_files": [{"split": "train", "path": "data/*/xho_Latn/*.parquet"}]}, {"config_name": "ydd_Hebr", "data_files": [{"split": "train", "path": "data/*/ydd_Hebr/*.parquet"}]}, {"config_name": "yor_Latn", "data_files": [{"split": "train", "path": "data/*/yor_Latn/*.parquet"}]}, {"config_name": "yue_Hani", "data_files": [{"split": "train", "path": "data/*/yue_Hani/*.parquet"}]}, {"config_name": "zsm_Latn", "data_files": [{"split": "train", "path": "data/*/zsm_Latn/*.parquet"}]}, {"config_name": "zul_Latn", "data_files": [{"split": "train", "path": "data/*/zul_Latn/*.parquet"}]}, {"config_name": "unlabelled", "data_files": [{"split": "train", "path": "unlabelled/*.parquet"}]}, {"config_name": "polygons", "data_files": [{"split": "train", "path": "polygons/*.parquet"}]}, {"config_name": "embeddings", "data_files": [{"split": "train", "path": "embeddings/*.parquet"}]}, {"config_name": "polygons_labels_shares", "data_files": [{"split": "train", "path": "polygons_labels_shares/*.parquet"}]}, {"config_name": "embeddings_en", "data_files": [{"split": "train", "path": "embeddings_en/*.parquet"}]}]
---

![GeoReSeT: collect texts linked to OpenStreetMap polygons, filter them by land-use relevance, link sentences back to polygons](assets/hero.png)

# GeoReSeT Corpus

**96,675,303 sentences in 95 languages, tied to 3,704,364 OpenStreetMap polygons in 386 regions, each labelled for land-use / land-cover relevance by an LLM.** English text comes with them: 72,421,077 sentences machine-translated with NLLB-200 and 25,524,307 already in English (see [English translations](#english-translations)).

Sentences come from the websites, `description` tags and Wikipedia / Wikivoyage articles linked to OpenStreetMap polygons. Every row is one sentence with a boolean `landuse` label, the language, and the location of the polygon it describes. Polygon shapes are in a separate `polygons` table.

**The full dataset**, all sentences:

![The full dataset per H3 cell](assets/map.png)

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("NoeFlandre/GeoReSeT-Corpus", "fra_Latn", split="train")  # one language
ds = load_dataset("NoeFlandre/GeoReSeT-Corpus", "wikipedia", split="train")  # one source
ds = load_dataset("NoeFlandre/GeoReSeT-Corpus", split="train")  # everything
afg = load_dataset(
    "NoeFlandre/GeoReSeT-Corpus", "wikipedia", split="train", filters=[("region", "==", "afghanistan")]
)  # one region
```

Join sentences to polygon shapes with DuckDB, reading only what the query needs:

```sql
SELECT s.text, s.landuse, p.osm_tag, p.geometry
FROM 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/data/wikipedia/*/*.parquet' s
JOIN 'hf://datasets/NoeFlandre/GeoReSeT-Corpus/polygons/*.parquet' p ON p.polygon_id = s.polygon_ids[1]
WHERE s.region = 'afghanistan' AND s.landuse;
```

English only, whatever the original language:

```python
english = load_dataset("NoeFlandre/GeoReSeT-Corpus", split="train").select_columns(["id", "text_en", "translation", "landuse"])
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
| wikipedia | 58,052,517 | 933,468,335 | 678,711 | 85 | 386 | 40.4% |
| website | 36,379,716 | 318,284,938 | 1,156,518 | 94 | 382 | 19.7% |
| wikivoyage | 1,280,056 | 19,195,698 | 5,801 | 26 | 297 | 48.6% |
| osm_description | 963,014 | 5,093,281 | 822,892 | 79 | 384 | 75.3% |
| **total** | 96,675,303 | 1,276,042,252 | 2,613,738 | 95 | 386 | 33.0% |

`polygons` counts distinct polygons linked to the sentences; the `polygons` table holds all 3,704,364. Words are whitespace-separated tokens, so Chinese, Japanese and Thai are undercounted.

![Sentences per language, top 20](assets/languages.png)

| Continent | Sentences | Polygons |
|---|---:|---:|
| europe | 55,158,209 | 1,449,445 |
| north-america | 20,336,824 | 555,832 |
| asia | 13,165,055 | 228,439 |
| south-america | 2,955,273 | 189,345 |
| oceania | 2,346,253 | 57,665 |
| africa | 2,143,242 | 121,316 |
| central-america | 495,980 | 10,859 |
| antarctica | 74,467 | 948 |

Polygon areas, over all 3,704,364 polygons:

| 10% | 25% | Median | 75% | 90% |
|---:|---:|---:|---:|---:|
| 84 m² | 239 m² | 917 m² | 1.0 ha | 8.5 ha |

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
| osm_description | 11,925 | 66,974 |
| website | 356,214 | 31,640 |
| wikipedia | 1,276,862 | 1,229,404 |
| wikivoyage | 29,787 | 144 |
| **total** | 1,674,788 | 1,328,162 |
## Principal components

Each sentence is turned into a list of 1,024 numbers, its embedding, so that sentences about similar things get similar lists. The maps show, place by place, which kinds of text are common, along three main directions, shown as red, green and blue.

### How it is computed

1. **Embed.** Each sentence becomes an embedding with [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (multilingual maps: 27,495,185 sentences).
2. **Average per place.** Sentences are grouped by [H3](https://h3geo.org) cell, about 1,800 km² each. Each cell takes the average of its sentences' embeddings, so every place counts once, however much is written there. Places with fewer than 20 sentences are left out.
3. **Find the directions.** [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) finds the three directions along which these place averages differ the most. Each direction is an axis, and each place gets a score on each axis.
4. **Colour.** A place's scores on axes 1, 2 and 3 become its red, green and blue, scaled between each axis' 2nd and 98th percentile. Places with similar colours talk about similar things.

**Variance** is the share of all the differences between places that an axis captures.

### Multilingual: all relevant sentences

Each place's sentences, in their original languages (60,959 places).

![Colour map of the relevant sentences: red, green, blue = axes 1, 2, 3](assets/map_pca.png)

| Axis | Variance |
|---|---:|
| 1 (red) | 11.9% |
| 2 (green) | 8.3% |
| 3 (blue) | 7.9% |

Each axis on its own:

![Axis 1 (red)](assets/map_pc1.png)
![Axis 2 (green)](assets/map_pc2.png)
![Axis 3 (blue)](assets/map_pc3.png)

### English: original and translated

The relevant sentences written in English, plus the [English translations](#english-translations) of the others, pooled (26,768,603 sentences, 60,509 places). Their embeddings come from the same model, so both kinds sit in one space.

![Colour map of the relevant sentences in English, original and translated: red, green, blue = axes 1, 2, 3](assets/map_pca_english_pool.png)

| Axis | Variance |
|---|---:|
| 1 (red) | 14.5% |
| 2 (green) | 10.7% |
| 3 (blue) | 6.7% |

Each axis on its own:

![Axis 1 (red)](assets/map_pc1_english_pool.png)
![Axis 2 (green)](assets/map_pc2_english_pool.png)
![Axis 3 (blue)](assets/map_pc3_english_pool.png)


## English translations

72,421,077 sentences were translated into English with [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) (58,366,000 distinct texts from 179 source languages, each translated once); the others are English already or were left out:

| `translation` | Sentences |
|---:|---:|
| `nllb` | 72,421,077 |
| `original` | 25,524,307 |
| `unsplit` | 1,302,548 |
| `unsupported` | 430,321 |

The model was chosen on [`llms-xx-to-en-translation`](https://huggingface.co/datasets/NoeFlandre/llms-xx-to-en-translation), a benchmark of 17 open models on the corpus' languages into English, where NLLB-200 3.3B reaches chrF++ 66.4 and COMET-22 0.889; the settings below are the benchmark's.

* **Source language.** The row's `language`, except for `website` sentences, which carry their page's language: when GlotLID v4 gives a sentence another language with probability ≥ 0.9, that language is used (an English menu line on a Hindi page is kept as it is). Chinese (`cmn_Hani`) is read as Traditional (`zho_Hant`) when it has more Traditional-only than Simplified-only characters, else Simplified; `fas_Arab`, `ekk_Latn` and `yue_Hani` are NLLB's `pes_Arab`, `est_Latn` and `yue_Hant`; Serbian in Latin script is transliterated to Cyrillic first. Latin, Western Frisian, Uzbek in Cyrillic and romanized Indic and Arabic text are not supported (`unsupported`).
* **Decoding.** Greedy, float16, with [CTranslate2](https://github.com/OpenNMT/CTranslate2); each sentence alone (no document context); output budget 3× the source tokens, between 32 and 2,048 as in the benchmark, capped at 1,020 because CTranslate2 has no decoder position past 1,023 (outputs reaching the cap are flagged `truncated`).
* **Checks.** `translation_flags` lists the benchmark's failure checks (empty output, repeated n-grams, output length outside 0.33–3× the source's in words, a copy of the source, an output GlotLID v4 does not label English), plus `truncated` (the output budget was reached) and `input_truncated` (source over 1,022 tokens, cut):

| Check | Sentences | Share of translated |
|---|---:|---:|
| length_ratio_outlier | 5,214,924 | 7.2% |
| wrong_language | 3,566,334 | 4.9% |
| untranslated_copy | 997,083 | 1.4% |
| truncated | 410,667 | 0.6% |
| repetition_loop | 264,970 | 0.4% |
| empty | 4,735 | 0.0% |
| input_truncated | 3,903 | 0.0% |

## EUNIS labels

`polygons_labels_shares` gives each polygon the share of each [EUNIS](https://zenodo.org/records/16985100) habitat class over the pixels of the EUNIS wall-to-wall map whose centre lies inside it (`eunis_code`, `eunis_shares`, `max_eunis_share_code`, `max_eunis_share_label`), with the polygon's fields; join on `polygon_id`. If no pixel centre lies inside, the pixel under the polygon's centroid is used (`label_method` = `centroid`; otherwise `polygon`), with shares `[1.0]`. `unlabelled_reason` gives why a polygon has no label. Labels are null outside the map's coverage and on nodata pixels: 2,020,859 of the 3,704,364 polygons have one. Copied from [`NoeFlandre/GeoReSeT-Corpus-Fused` @ `ae05227`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus-Fused/tree/ae0522702a335b435be17fee3a5373bf26d25b52), as are the relevant sentences' vectors in `embeddings` (see [Methodology](#methodology)).

## Files

```
data/<source>/<language>/train-*.parquet   labelled sentences, sorted by region
unlabelled/train-*.parquet                 sentences without a label (see above)
polygons/train-*.parquet                   polygon shapes (GeoParquet)
embeddings/train-*.parquet                 relevant sentences, one vector each (see [Methodology](#methodology))
polygons_labels_shares/train-*.parquet     one row per polygon, with its EUNIS label and class shares (see [EUNIS labels](#eunis-labels))
embeddings_en/train-*.parquet              English translations of the relevant sentences, one vector each (see [Methodology](#methodology))
```

Configs: `all` (default), one per source, one per language, `unlabelled`, `polygons`, `embeddings`, `polygons_labels_shares`, `embeddings_en`.

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
   * [`NoeFlandre/osm-polygon-website-tag-landuse` @ `e6524c7`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag-landuse/tree/e6524c711ee0033c3cd5990e5fdbbe4b89992dfb)
   * [`NoeFlandre/osm-polygon-description-tag-landuse` @ `ce98b32`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-landuse/tree/ce98b328f3c6543800a3d931877b51eb3f0cf768)
   * [`NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` @ `49307ae`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse/tree/49307aeb05b6dcc8a5d393f77f59a1e6d13a967a)
6. **This dataset.** One row per sentence of each document (a document linked to several polygons appears once); overlapping extracts resolved to the most specific one; language codes normalized; sentences with fewer than two letters dropped, and Wikipedia / Wikivoyage rows made only of bracketed edit links or citations (`[ edit | edit source ]`, `[Smith 1998]`) removed. Join the raw outputs on `sha256(text)` = `text_sha256`. Code: https://github.com/NoeFlandre/GeoReSeT-Corpus

7. **English translations.** NLLB-200 3.3B ([revision `1a07f7d`](https://huggingface.co/facebook/nllb-200-3.3B/tree/1a07f7d195896b2114afcb79b7b57ab512e7b43e)) through CTranslate2 on [Grid'5000](https://www.grid5000.fr) GPUs, with the settings of [`llms-xx-to-en-translation`](https://github.com/NoeFlandre/llms-xx-to-en-translation) (see [English translations](#english-translations)). Code: https://github.com/NoeFlandre/GeoReSeT-Corpus8. **Embeddings of the relevant sentences.** The 27,495,185 distinct sentences with `landuse` true (the first `id` keeps a repeated text) embedded with [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (sentence-transformers, bf16, instruction `Instruct: Identify the land use or land cover described\nQuery: `, last-token pooling, L2-normalised), stored as float16 in `embeddings` with their `id`, `lat` and `lon`; join on `id` for the rest. For the maps, each counts in the H3 cell of its location (resolution 5), each cell takes the mean of its texts, cells with fewer than 20 sentences are left out, and PCA is fitted on the means of the other cells (each cell counts once). Maps use the Equal Earth projection.
9. **Embeddings of the English translations.** The 21,721,763 distinct `text_en` of the relevant sentences that NLLB-200 translated (`translation` = `nllb`; the smallest `id` keeps a repeated translation) embedded like `embeddings`, in the same space: [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (sentence-transformers, bf16, or float32 on GPUs without bf16; instruction `Instruct: Identify the land use or land cover described\nQuery: `, last-token pooling, at most 512 tokens, L2-normalised), stored as float16 in `embeddings_en` with their `id`, `lat` and `lon`; join on `id` for the rest. Relevant sentences already in English (`translation` = `original`) have theirs in `embeddings`. On [Grid'5000](https://www.grid5000.fr) GPUs. Code: https://github.com/NoeFlandre/GeoReSeT-Corpus
## License

* OpenStreetMap data (`osm_description` text, polygons, names, tags): [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors.
* Wikipedia and Wikivoyage text: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); `url` points to the exact revision for attribution.
* Website text (`license = other`): © the respective site owners, included for research. To have content removed, open a discussion on this repository.

* `text_en` of a `nllb` row is a machine translation of `text` and keeps `text`'s license. NLLB-200's weights are CC-BY-NC 4.0; the translations were produced for research.