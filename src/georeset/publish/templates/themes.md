<!-- section -->


**What places talk about, from the $themes_what**: each is embedded with [`$themes_model`](https://huggingface.co/$themes_model) and counted $themes_placement, and the mean embedding of each of the $themes_hexagons 2° hexagons holding $themes_min $themes_unit or more is projected on its first three principal components, shown as red, green and blue:

![Mean embedding of the $themes_what per 2° hexagon, first three principal components as red, green and blue](assets/map_pca$themes_suffix.png)

$themes_legend

The same axes and colours on $themes_fine_degrees° hexagons holding $themes_fine_min $themes_unit or more:

![Mean embedding of the $themes_what per $themes_fine_degrees° hexagon, coloured as the map above](assets/map_pca_fine$themes_suffix.png)

Each axis on its own, blue below zero and orange above. The $themes_what at both ends of each axis are in [`$themes_report`]($code_url/blob/master/docs/$themes_report).

![First principal component of the $themes_what per 2° hexagon](assets/map_pc1$themes_suffix.png)

![Second principal component of the $themes_what per 2° hexagon](assets/map_pc2$themes_suffix.png)

![Third principal component of the $themes_what per 2° hexagon](assets/map_pc3$themes_suffix.png)
<!-- files -->

$themes_files$themes_what, one vector each (see [Methodology](#methodology))
<!-- configs -->
, `$themes_folder`
<!-- method -->
$themes_step. **Embeddings of the $themes_what.** The $themes_texts $themes_selection embedded with [`$themes_model`](https://huggingface.co/$themes_model) (sentence-transformers, bf16, instruction `$themes_prompt`, last-token pooling, L2-normalised), stored as float16 in `$themes_folder` with their `id`, `lat` and `lon`; join on `id` for the rest. For the maps, each counts $themes_placement in matplotlib's 2° hexagon grid, hexagons with fewer than $themes_min $themes_unit dropped, and PCA fitted on the hexagon means (each hexagon counts once).
