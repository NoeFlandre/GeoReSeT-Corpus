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

Labelled sentences are rows of `data/`; `failed` and `skipped_unsplit` go to `unlabelled` with `status` and
`failure_reason`. The labels are not in `data/`: they live in the labelling round's sidecar (see Labelling rounds).
Partitions are sorted by region, then document and position, so region filters read few row groups.
Staged and intermediate files are deleted as soon as they are consumed.

## Labelling rounds

A round is one labelling of the corpus: a model, a draft, a prompt and the parser that reads its output.
Its name is the model slug and the prompt's sha256, e.g. `lfm25-dspark-p2fb4856`.

```
labels/rounds.json                         the registry: each round's configuration and code commits
labels/<round>/decisions.parquet           id, text_sha256, decision (yes / no / failed), failure
labels/<round>/generations-*.parquet       group, text_sha256, raw_output, finish_reason, truncated (4 shards)
labels/<round>/agreement.json              the parser's agreement with the round's decisions
```

A row's label is its `decision` in a round, joined on `id`. The build works in 16 buckets (the first hex digit of
`text_sha256`) and publishes decisions as one file and generations in four shards.
Round 1 is the corpus's original labels, with their raw outputs from the three filter repositories.
`label.attach.landuse` gives any pipeline the relevant flag: the release's `landuse` column before v2.0, the current
round's `yes` decisions from v2.0 on.

* Build (CPU, whole node): `label-round-build` joins the corpus to the generations and writes the round's folder.
* Propose: `label-round-propose` opens a pull request with `data/`, `labels/` and the card.
* Finalize, after review: `hub.release` merges the pull request and tags the release.

A release never removes a round already on the Hub: `labels/` is added to, and the registry keeps every round.

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

## What places talk about

Three PCA maps added to the fused card: the relevant sentences (`--corpus landuse`, the default), the same
sentences in English only (`--corpus english`: same vectors, no language effect), and the scene descriptions
(`--corpus scenes`):

```
georeset themes-sentences --corpus C --sentences S/       distinct texts: 128 shards (landuse), 1 (scenes)
uv run scripts/embed.py S/ --folder F --worker k --workers n   on GPUs; each shard → F/ on the Hub
georeset themes --corpus C --work work-fused --sentences S/    cell means, PCA, axis ends → work-fused/themes/C/
georeset fuse ...                                         card: maps, legends
```

`F` is `embeddings` for the sentences and `scene_embeddings` for the scenes. `scripts/embed.py` is
self-contained (PEP 723 dependencies): `microsoft/harrier-oss-v1-0.6b` in bf16, the instruction
`Identify the land use or land cover described`, L2-normalised vectors stored as float16. Shards already on
the Hub are skipped, so jobs can be killed, resubmitted or run on several GPUs at once.
Maps use the Equal Earth projection (`georeset.publish.geo`) and a single H3 resolution, 5 (about 250 km²):
every count and every map is per resolution-5 cell. Each text counts once, in its own cell (a scene
description counts in every cell where its scene has photos), and each cell takes the mean embedding of its
texts; those with fewer than 20 sentences (10 scene descriptions) are dropped and PCA is fitted on the rest.
The two passes over the shards (cells, cell means) load the next shards in the background. The card shows
the PCA only: its maps and each component's share of variance, with no axis names or axis reports.

## Explorer Space

[An interactive map](https://huggingface.co/spaces/NoeFlandre/GeoReSeT-Corpus-Fused) of the fused dataset,
on the same two layers: numbers per H3 cell of resolution 5, each cell on its own (no zoom levels, no coarser
cells, nothing averaged over other cells). The PCA colours are each resolution-5 cell's own colour. It shows only
the sentences the labelling round decides are relevant
(`decision` yes), from every source or one; layers: those sentences, scene photos, their most frequent
language and the three PCA maps; a click shows a cell's numbers, up to three relevant sentences and the
scene with the most photos there. The dataset itself is not changed.

```
georeset themes --corpus C ...   (each corpus, as above: also writes every resolution-5 cell's scores)
georeset space                   data from the dataset's current revision, app files → the Space
```

`georeset.publish.space` writes the data (`r5/` and `samples/` split by resolution-1
parent, loaded as the map moves) and `manifest.json`, which names the dataset revision it was built from.
The app (`georeset/publish/space_app`, a static page: MapLibre, deck.gl, h3-js, hyparquet) shows that
revision and says when the dataset has changed since: rebuild with the two steps above after each release.

## English translations

`georeset.translate` adds four columns to every row of `data/` and `unlabelled/` of a release
(`text_en`, `translation`, `translation_language`, `translation_flags`) without changing any other column,
with [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) and the settings of the benchmark
that chose it ([`llms-xx-to-en-translation`](https://github.com/NoeFlandre/llms-xx-to-en-translation): greedy,
float16, output budget 3× the source tokens in [32, 2048] capped at 1,020 decoder positions, its failure checks, GlotLID v4). Everything runs on
Grid'5000; working files live in the private bucket `NoeFlandre/georeset-translation`, deleted at the end.

```
release v1.0 ──translate-plan──▶ bucket work/<unit>/<n>.parquet   distinct (language, text) to translate
   (CPU job)                     bucket routes/<unit>.parquet       website rows GlotLID reassigns
                                 bucket plan/<unit>.json            counts
             ──translate-work──▶ bucket results/<unit>/<n>.parquet  one GPU job per free GPU (g5k-dispatch);
   (GPU jobs)                                                       work shards deleted once translated
             ──translate-assemble──▶ out/ (release + 4 columns, original columns checked equal), card
   (CPU job)
             ──translate-assemble─▶ a pull request on the dataset (not polygons); translate-finalize: merge, tag v1.1, delete the bucket
```

A unit is one language folder (all sources) or `unlabelled`. Routing (`translate.route`): English rows keep
their text; `website` sentences take GlotLID's label when it is another language with probability ≥ 0.9
(pages are labelled as a whole); Chinese is split into Simplified and Traditional by distinctive characters;
`fas_Arab`, `ekk_Latn`, `yue_Hani` map to NLLB's codes; Serbian Latin is transliterated to Cyrillic;
languages NLLB lacks are `unsupported`, unsplit texts `unsplit`. Each distinct text is translated once, keyed
by `sha256(code, text)[:16]`; the assembly routes rows again with the same code and the recorded overrides.

```bash
# on a Nancy frontend, once
oarsub -n georeset-convert -p "gpu_count=0 AND memnode>=98304" -l nodes=1,walltime=1:30 \
  "bash ~/src/GeoReSeT-Corpus/scripts/g5k/convert_model.sh"
oarsub -n georeset-plan -p "gpu_count=0" -l nodes=1,walltime=6:00 "bash ~/src/GeoReSeT-Corpus/scripts/g5k/plan_job.sh"
# GPU jobs on every site until no work shard is left
oarsub -n dispatch -p "gpu_count=0" -l nodes=1,walltime=1:00 \
  "bash ~/src/g5k-dispatch/scripts/dispatch_job.sh ~/src/GeoReSeT-Corpus/configs/translate.yaml"
# then
oarsub -n georeset-assemble -p "gpu_count=0 AND memnode>=98304" -l nodes=1,walltime=4:00 "bash ~/src/GeoReSeT-Corpus/scripts/g5k/assemble_job.sh"
georeset translate-finalize --pr <number>   # after reviewing the pull request the assembly opened
```
