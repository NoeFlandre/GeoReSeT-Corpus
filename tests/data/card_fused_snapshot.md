---
license: ["odbl", "cc-by-sa-4.0", "cc-by-4.0", "other"]
pretty_name: "GeoReSeT Corpus Fused"
language: ["fra"]
task_categories: ["text-classification"]
tags: ["openstreetmap", "land-use", "land-cover", "geospatial", "multilingual", "street-level-imagery", "scene-description"]
size_categories: ["n<1K"]
configs: [{"config_name": "all", "data_files": [{"split": "train", "path": "data/*/*/*.parquet"}], "default": true}, {"config_name": "wikipedia", "data_files": [{"split": "train", "path": "data/wikipedia/*/*.parquet"}]}, {"config_name": "fra_Latn", "data_files": [{"split": "train", "path": "data/*/fra_Latn/*.parquet"}]}, {"config_name": "unlabelled", "data_files": [{"split": "train", "path": "unlabelled/*.parquet"}]}, {"config_name": "polygons", "data_files": [{"split": "train", "path": "polygons/*.parquet"}]}, {"config_name": "scenes", "data_files": [{"split": "train", "path": "scenes/*.parquet"}]}, {"config_name": "scene_photos", "data_files": [{"split": "train", "path": "scene_photos/*.parquet"}]}]
---

![GeoReSeT Fused: texts linked to OpenStreetMap polygons are filtered by land-use relevance; geotagged photos are clustered and described by a vision-language model; both are tied back to places](assets/hero.png)

# GeoReSeT Corpus Fused

**10 sentences in 1 languages, tied to 7 OpenStreetMap polygons in 2 regions, each labelled for land-use / land-cover relevance by an LLM.**

**Plus 2 English scene descriptions, each labelled with a land-use / land-cover class by a vision-language model, tied to the locations of the 1,500 geotagged photos they describe.**

Sentences come from the websites, `description` tags and Wikipedia / Wikivoyage articles linked to OpenStreetMap polygons. Every row is one sentence with a boolean `landuse` label, the language, and the location of the polygon it describes. Polygon shapes are in a separate `polygons` table. Scene descriptions come from ground-level photos (Mapillary, Flickr, Google Landmarks v2, iNaturalist, KartaView, camera traps) grouped by look with [geo-rag](https://github.com/ananthu-aniraj/geo-rag); see [Scenes](#scenes). No photo is included.

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

Every photo location with the description and label of its scene, or one row per description:

```python
photos = load_dataset("me/data", "scene_photos", split="train")  # text, lulc_label, lat, lon, ...
scenes = load_dataset("me/data", "scenes", split="train")
local = scenes.filter(lambda s: s["spread_km_p90"] <= 100)  # descriptions of one area
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
scenes/train-*.parquet                     scene descriptions, one row per scene
scene_photos/train-*.parquet               one row per photo, with its scene's text and label
```

Configs: `all` (default), one per source, one per language, `unlabelled`, `polygons`, `scenes`, `scene_photos`.

## Unlabelled sentences

The `unlabelled` config holds sentences without a usable label, kept to be labelled later. It has the same fields, with `status` and `failure_reason` instead of `landuse`:

* `failed`: the model gave no usable yes/no answer (`failure_reason`, nearly always `truncated`).
* `unsplit`: text the upstream splitter could not segment (unsupported language); `text` is a whole page, description or paragraph.

| Source | `failed` | `unsplit` |
|---|---:|---:|
| wikipedia | 1 | 2 |
| **total** | 1 | 2 |

## Scenes

2 descriptions (40 words) of 1,500 geotagged ground-level photos in 3 countries, from [geo-rag](https://github.com/ananthu-aniraj/geo-rag) by Ananthu Aniraj. Photos were grouped by look into clusters (median 750 photos); a vision-language model described 4 representative photos of each cluster and labelled the scene with one of 40 land-use / land-cover classes.

A description belongs to all the photos of its cluster, and a cluster groups photos that **look** alike, not photos of one place: half the clusters have photos in 20 countries or more. So:

* `scene_photos` repeats the text and label of the scene on each of its photos, with the photo's own location, time and climate: one row per (text, place).
* `scenes` holds each text once, with how far its photos are spread. Filter on `spread_km_p90` for descriptions of one area.

Scenes whose photos lie, for 90% of them, within a distance of their median location:

| 10 km | 100 km | 1,000 km |
|---:|---:|---:|
| 0 | 1 | 2 |

Most frequent labels:

| Label | Scenes | Photos |
|---|---:|---:|
| Tundra | 1 | 1,000 |
| Lake / Pond | 1 | 500 |

Photos per platform, and the share with a public `url`:

| Platform | Photos | With `url` |
|---|---:|---:|
| flickr | 1,500 | 25.0% |

| Field (`scene_photos`) | Type | Description |
|---|---|---|
| `scene_id` | int | Scene (cluster) of the photo |
| `text` | string | Description of the scene (English, generated) |
| `lulc_label` | string | Land-use / land-cover class of the scene (generated) |
| `lat`, `lon` | float | Location of the photo (WGS 84) |
| `h3_cell` | string | [H3](https://h3geo.org) cell of the photo, resolution 11 |
| `country`, `continent` | string | Where the photo lies ([Natural Earth](https://www.naturalearthdata.com) borders) |
| `captured_at` | string | Capture time (UTC, ISO 8601) |
| `season`, `time_of_day` | string | Climate-aware season (`Wet Season`, `Summer`, ...) and local time slot (`Dawn`, ...) |
| `koppen` | string | [Köppen-Geiger](https://www.gloh2o.org/koppen/) climate zone (`Cfb`) |
| `platform`, `photo_id` | string | Where the photo comes from, and its id there |
| `url` | string | Public link to the photo; empty when its license forbids reuse or it is not online |
| `photo_license` | string | License of the photo, as given by its platform |

The `scenes` config has one row per description: `scene_id`, `text`, `visual_description` (the model's first-pass description of the photos, without location context), `lulc_label`, `parent_scene_id` and `parent_lulc_label` (one of 2,000 coarser clusters), `n_photos`, `lat` / `lon` (median of its photos), `spread_km_p50` / `spread_km_p90` (distance from that median within which 50% / 90% of its photos lie) and `countries` (most frequent first).

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
7. **Scenes** ([geo-rag](https://github.com/ananthu-aniraj/geo-rag) @ [`4252e04`](https://github.com/ananthu-aniraj/geo-rag/tree/4252e043b3e8f1afbbc1a6cfb55e3577bcb4ee04)). Geotagged photos scraped from Mapillary, Flickr, iNaturalist and KartaView, plus Google Landmarks v2 and camera-trap archives; duplicates removed on [H3](https://h3geo.org) cells and GPS glitches filtered. Photos embedded with [`google/tipsv2-b14`](https://huggingface.co/google/tipsv2-b14) (CLS) and clustered with FAISS k-means into 40,000 clusters under 2,000 parents. For each cluster, [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) described a strip of 4 diverse photos near its centre (`visual_description`), then, given that description and the photos' location, season and Köppen climate, chose a land-use / land-cover class and wrote the final description (`text`). Here: the 2 clusters whose labelling failed are dropped, 6 labels outside the class list mapped to the nearest one, local file paths removed, Mapillary keys turned into links, and links removed for photos whose license forbids reuse. Raw input: [`me/raw` @ `fffffff`](https://huggingface.co/datasets/me/raw/tree/ffffffffffffffffffffffffffffffffffffffff) (private).

## License

* OpenStreetMap data (`osm_description` text, polygons, names, tags): [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors.
* Wikipedia and Wikivoyage text: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); `url` points to the exact revision for attribution.
* Website text (`license = other`): © the respective site owners, included for research. To have content removed, open a discussion on this repository.
* Scene descriptions (`text`, `visual_description`, `lulc_label`): generated by Gemma 4 (Apache 2.0), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* Scene photos: no photo is redistributed. Each row gives the photo's own license in `photo_license` (Mapillary and KartaView: CC BY-SA 4.0; iNaturalist: CC BY-NC 4.0; Flickr and Wikimedia Commons: per photo). Photos marked `All Rights Reserved` or `Do not reuse` keep only their id and location, with no `url`.