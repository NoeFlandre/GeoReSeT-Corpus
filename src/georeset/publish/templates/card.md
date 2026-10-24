---
$front_matter
---

![GeoReSeT: collect texts linked to OpenStreetMap polygons, filter them by land-use relevance, link sentences back to polygons](assets/hero.png)

# GeoReSeT Corpus

**$sentences sentences in $languages languages, tied to $polygons OpenStreetMap polygons in $regions regions, each labelled for land-use / land-cover relevance.**

Sentences come from the websites, `description` tags and Wikipedia / Wikivoyage articles linked to OpenStreetMap polygons. Every row is one sentence with a boolean `landuse` label, the language, and the location of the polygon it describes. Polygon shapes are in a separate `polygons` table.

![Sentences per H3 cell](assets/map.png)

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("$repo_id", "$example_language", split="train")  # one language
ds = load_dataset("$repo_id", "wikipedia", split="train")  # one source
ds = load_dataset("$repo_id", split="train")  # everything
afg = load_dataset(
    "$repo_id", "wikipedia", split="train", filters=[("region", "==", "afghanistan")]
)  # one region
```

Join sentences to polygon shapes with DuckDB, reading only what the query needs:

```sql
SELECT s.text, s.landuse, p.osm_tag, p.geometry
FROM 'hf://datasets/$repo_id/data/wikipedia/*/*.parquet' s
JOIN 'hf://datasets/$repo_id/polygons/*.parquet' p ON p.polygon_id = s.polygon_ids[1]
WHERE s.region = 'afghanistan' AND s.landuse;
```

## Contents

$by_source

`polygons` counts distinct polygons linked to the sentences; the `polygons` table holds all $polygons. Words are whitespace-separated tokens, so Chinese, Japanese and Thai are undercounted.

![Sentences per language, top 20](assets/languages.png)

$continents

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable id of the sentence (128-bit hex) |
| `text` | string | The sentence, byte-exact as labelled |
| `landuse` | bool | Relevant to land use / land cover |
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

* `website`: [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) on the page text.
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

$unlabelled

## How it was made

1. OpenStreetMap polygons with a website, a `description` tag, or a Wikidata link to Wikipedia / Wikivoyage were extracted per [Geofabrik](https://download.geofabrik.de) region.
2. Texts were split into sentences and each sentence labelled by `LiquidAI/LFM2.5-2.6B` for land-use / land-cover relevance. Labels, method and quality checks are documented in the source datasets:
$sources
3. This dataset merges them: one row per sentence of each document (a document linked to several polygons appears once), overlapping regions resolved to the most specific extract, language codes normalized, sentences with fewer than two letters dropped.

The model's raw outputs, including its reasoning, stay in the source datasets: join them on `sha256(text)` = `text_sha256`.

Code: $code_url

## License

* OpenStreetMap data (`osm_description` text, polygons, names, tags): [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors.
* Wikipedia and Wikivoyage text: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); `url` points to the exact revision for attribution.
* Website text (`license = other`): © the respective site owners, included for research. To have content removed, open a discussion on this repository.
