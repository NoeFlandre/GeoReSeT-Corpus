# Pipeline

Code layout:

```
georeset/
  cli.py        command line
  hub.py        Hugging Face Hub I/O
  normalize/    languages, regions, OSM helpers (pure)
  sources/      one adapter per upstream repo + shared base; scenes (geo-rag)
  build/        stage, assemble, pipeline, write; fuse
  publish/      stats, geo (map maths), figures (matplotlib), card rendering, report
```

```
upstream repos ──stage──▶ work/stage/<source>/<region>/{polygons,links,sentences}.parquet
               ──meta───▶ work/meta/polygons.parquet, work/meta/docs/<region>.parquet
               ──rows───▶ work/rows/data/<source>/<language>/*.parquet, work/rows/unlabelled/
               ──shards─▶ out/data/<source>/<language>/train-*.parquet, out/unlabelled/, out/polygons/
               ──card───▶ out/README.md, out/assets/
```

## Stage

Each source adapter (`georeset.sources.website`, `.description`, `.wiki`) turns one region of one upstream repo into three common tables:

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

## Fuse

`georeset fuse` builds the fused dataset without rebuilding the corpus:

```
corpus repo @ v1.0 ──▶ out-fused/{data,unlabelled,polygons}/   (copied as published)
geo-rag-scenes-raw ──▶ out-fused/scenes/, out-fused/scene_photos/
                   ──▶ out-fused/README.md, out-fused/assets/  (card with the fused hero)
```

The raw geo-rag files are pinned in the private repo `NoeFlandre/geo-rag-scenes-raw`.
`georeset.sources.scenes` keeps one row per described cluster (failed ones dropped, off-vocabulary labels
mapped), one row per photo (local paths and links to non-reusable photos removed, Mapillary keys turned
into links), and how far each scene's photos are spread. `scene_photos` repeats each scene's text on its
photos, sorted by scene so the text compresses to one copy per row group.
