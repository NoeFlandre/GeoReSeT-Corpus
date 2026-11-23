<!-- section -->


**What places talk about, from the $themes_what**: each is embedded with [`$themes_model`](https://huggingface.co/$themes_model) and counts $themes_placement (resolution $themes_resolution). Each of the $themes_hexagons cells of resolution $themes_resolution holding $themes_min $themes_unit or more takes the mean embedding of its texts. Those means are projected on their first three principal components, shown as red, green and blue:

![Mean embedding of the $themes_what per H3 cell of resolution $themes_resolution, first three principal components as red, green and blue](assets/map_pca$themes_suffix.png)

$themes_legend

Each axis on its own, blue below zero and orange above.

![First principal component of the $themes_what per H3 cell of resolution $themes_resolution](assets/map_pc1$themes_suffix.png)

![Second principal component of the $themes_what per H3 cell of resolution $themes_resolution](assets/map_pc2$themes_suffix.png)

![Third principal component of the $themes_what per H3 cell of resolution $themes_resolution](assets/map_pc3$themes_suffix.png)
<!-- files -->

$themes_files$themes_what, one vector each (see [Methodology](#methodology))
<!-- configs -->
, `$themes_folder`
<!-- method -->
$themes_step. **Embeddings of the $themes_what.** The $themes_texts $themes_selection embedded with [`$themes_model`](https://huggingface.co/$themes_model) (sentence-transformers, bf16, instruction `$themes_prompt`, last-token pooling, L2-normalised), stored as float16 in `$themes_folder` with their `id`, `lat` and `lon`; join on `id` for the rest. For the maps, each counts $themes_placement (resolution $themes_resolution), each cell takes the mean of its texts, cells with fewer than $themes_min $themes_unit are left out, and PCA is fitted on the means of the other cells (each cell counts once). Maps use the Equal Earth projection.
