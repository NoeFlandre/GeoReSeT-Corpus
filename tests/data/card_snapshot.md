---
license: ["odbl", "cc-by-sa-4.0", "other"]
pretty_name: "GeoReSeT Corpus"
language: ["fra"]
task_categories: ["text-classification"]
tags: ["openstreetmap", "land-use", "land-cover", "geospatial", "multilingual"]
size_categories: ["n<1K"]
configs: [{"config_name": "all", "data_files": [{"split": "train", "path": "data/*/*/*.parquet"}], "default": true}, {"config_name": "wikipedia", "data_files": [{"split": "train", "path": "data/wikipedia/*/*.parquet"}]}, {"config_name": "fra_Latn", "data_files": [{"split": "train", "path": "data/*/fra_Latn/*.parquet"}]}, {"config_name": "unlabelled", "data_files": [{"split": "train", "path": "unlabelled/*.parquet"}]}, {"config_name": "polygons", "data_files": [{"split": "train", "path": "polygons/*.parquet"}]}]
---

![GeoReSeT: collect texts linked to OpenStreetMap polygons, filter them by land-use relevance, link sentences back to polygons](assets/hero.png)

# GeoReSeT Corpus

**10 sentences in 1 languages, tied to 7 OpenStreetMap polygons in 2 regions, each labelled for land-use / land-cover relevance by an LLM.**

Sentences come from the websites, `description` tags and Wikipedia / Wikivoyage articles linked to OpenStreetMap polygons. Every row is one sentence with a boolean `landuse` label, the language, and the location of the polygon it describes. Polygon shapes are in a separate `polygons` table.

![Sentences per H3 cell](assets/map.png)

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

## Contents

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

## Fields

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

The `polygons` config holds one row per polygon: `polygon_id`, `osm_type`, `osm_id`, `name`, `osm_tag`, `region`, `continent`, `lat`, `lon`, `area_m2` and `geometry` (WKB, [GeoParquet](https://geoparquet.org) 1.1, WGS 84).

## Languages

Labels follow GlotLID: ISO 639-3 + ISO 15924 script. How the language was found depends on the source:

* `website`: [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) on the whole page, so a sentence carries its page's label (an English menu line on a Hindi page is `hin_Deva`).
* `osm_description`: [Lingua](https://github.com/pemistahl/lingua-py), with GlotLID v3 as fallback.
* `wikipedia`, `wikivoyage`: the language edition of the article (`fr.wikipedia.org` → `fra_Latn`), not detected.

Only text in the 85 languages of the [SaT](https://huggingface.co/segment-any-text/sat-3l-sm) sentence splitter was split and labelled.
Labels are finer than that: individual languages of a macrolanguage (`nob_Latn` and `nno_Latn` for Norwegian, `cmn_Hani` and `yue_Hani` for Chinese)
and scripts (`srp_Cyrl` and `srp_Latn`) stay distinct. Other languages appear only in `unlabelled`, as `unsplit`.

## Files

```
data/<source>/<language>/train-*.parquet   labelled sentences, sorted by region
unlabelled/train-*.parquet                 sentences without a label (see below)
polygons/train-*.parquet                   polygon shapes (GeoParquet)
```

Configs: `all` (default), one per source, one per language, `unlabelled`, `polygons`.

## Unlabelled sentences

The `unlabelled` config holds sentences without a usable label, kept to be labelled later. It has the same fields, with `status` and `failure_reason` instead of `landuse`:

* `failed`: the model gave no usable yes/no answer (`failure_reason`, nearly always `truncated`).
* `unsplit`: text the upstream splitter could not segment (unsupported language); `text` is a whole page, description or paragraph.

| Source | `failed` | `unsplit` |
|---|---:|---:|
| wikipedia | 1 | 2 |
| **total** | 1 | 2 |

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

## License

* OpenStreetMap data (`osm_description` text, polygons, names, tags): [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors.
* Wikipedia and Wikivoyage text: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); `url` points to the exact revision for attribution.
* Website text (`license = other`): © the respective site owners, included for research. To have content removed, open a discussion on this repository.
