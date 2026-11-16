<!-- files -->

embeddings/train-*.parquet                 relevant sentences, one vector each (see [Methodology](#methodology))
polygons_labels_shares/train-*.parquet     one row per polygon, with its EUNIS label and class shares (see [EUNIS labels](#eunis-labels))
<!-- configs -->
, `embeddings`, `polygons_labels_shares`
<!-- section -->

## EUNIS labels

`polygons_labels_shares` gives each polygon the share of each [EUNIS](https://zenodo.org/records/16985100) habitat class over the pixels of the EUNIS wall-to-wall map whose centre lies inside it (`eunis_code`, `eunis_shares`, `max_eunis_share_code`, `max_eunis_share_label`), with the polygon's fields; join on `polygon_id`. If no pixel centre lies inside, the pixel under the polygon's centroid is used (`label_method` = `centroid`; otherwise `polygon`), with shares `[1.0]`. `unlabelled_reason` gives why a polygon has no label. Labels are null outside the map's coverage and on nodata pixels: $labelled of the $polygons polygons have one. Copied from $fused.
<!-- method -->

$step. **Embeddings of the relevant sentences.** The $embedded $selection embedded with [`$model`](https://huggingface.co/$model) (sentence-transformers, bf16, instruction `$prompt`, last-token pooling, L2-normalised), stored as float16 in `embeddings` with their `id`, `lat` and `lon`; join on `id` for the rest. Copied from $fused.
