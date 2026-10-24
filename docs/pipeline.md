# Pipeline

```
upstream repos ──stage──▶ work/stage/<source>/<region>/{polygons,links,sentences}.parquet
               ──meta───▶ work/meta/polygons.parquet, work/meta/docs/<region>.parquet
               ──rows───▶ work/rows/data/<source>/<language>/*.parquet, work/rows/unlabelled/
               ──shards─▶ out/data/<source>/<language>/train-*.parquet, out/unlabelled/, out/polygons/
               ──card───▶ out/README.md, out/assets/
```

## Stage

Each source adapter (`georeset.sources.*`) turns one region of one upstream repo into three common tables:

* **polygons**: `polygon_id` (`way/123`), name, primary OSM tag, centroid, area, WKB geometry.
* **links**: document → polygon. A document is a website URL, an OSM description, or a Wikipedia / Wikivoyage article.
* **sentences**: document, position, text, label, failure reason, language (normalized to `fra_Latn` style), URL.

Downloads run in parallel; conversions are capped to bound memory. Each region is written atomically and the download deleted.

## Meta

* Extracts overlap (`italy` and `centro`, `ukraine` and `russia-crimean-fed-district`, borders).
  Each polygon keeps the smallest extract containing it, and the best attributes (Wikipedia > website > description).
* A document staged in several regions is taken once, from its highest-priority region.
* Each document gets its sorted `polygon_ids`; the first one gives the row's region, location, name, tag and area.

## Rows and shards

Labelled (`yes` / `no`) sentences become `landuse = true / false`; `failed` and `skipped_unsplit`
go to `unlabelled` with `status` and `failure_reason`. Partitions are sorted by region, then document and position,
so region filters read few row groups. Staged and intermediate files are deleted as soon as they are consumed.
