# GeoReSeT Corpus

Code that builds and publishes [`NoeFlandre/GeoReSeT-Corpus`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus):
multilingual sentences tied to OpenStreetMap polygons, each labelled for land-use / land-cover relevance.

It merges three labelled source datasets into one clean, sentence-level dataset:

| Source dataset | Text |
|---|---|
| `osm-polygon-website-tag-landuse` | websites tagged on polygons |
| `osm-polygon-description-tag-landuse` | OSM `description*` tags |
| `osm-polygon-wikidata-and-wikipedia-landuse` | Wikipedia and Wikivoyage articles |

## Run it

```bash
uv sync
uv run georeset run --publish      # download → build → card → upload
```

Each step is also a command, and every step resumes where it stopped:

```bash
uv run georeset stage    # download each (source, region), convert, delete the download
uv run georeset meta     # one row per polygon, one copy per document
uv run georeset rows     # final rows per region, by source and language
uv run georeset shards   # sorted parquet shards in out/
uv run georeset card     # stats, map, language chart, README.md
uv run georeset publish  # upload out/ to the Hub
```

`--region italy-latest` (repeatable) restricts a run to some regions.

## Fused dataset

[`NoeFlandre/GeoReSeT-Corpus-Fused`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus-Fused)
adds the scene descriptions of [geo-rag](https://github.com/ananthu-aniraj/geo-rag) to a corpus release:

```bash
uv run georeset fuse --publish     # copy corpus v1.0, add scenes, card, upload
```

| Config | Rows |
|---|---|
| `scenes` | one per scene description (a cluster of similar-looking photos) |
| `scene_photos` | one per photo: its scene's text and label with the photo's location |
