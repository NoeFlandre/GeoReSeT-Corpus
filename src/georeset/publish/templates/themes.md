<!-- section -->


**What places talk about, from the $themes_what**: each is embedded with [`$themes_model`](https://huggingface.co/$themes_model) and counts $themes_placement (resolution $themes_resolution). Each cell takes the mean embedding of its texts, and each of the $themes_hexagons cells of resolution $themes_display holding $themes_min $themes_unit or more the mean of its child cells, so that every place counts once, however much is written there. Those means are projected on their first three principal components, shown as red, green and blue. Within that colour, a cell is a little darker or lighter, and its hue a little turned, by how it differs from its surroundings: a second PCA, of each cell's mean minus its region's (resolution 3, about 12,000 km²), whose two axes are listed last:

![Mean embedding of the $themes_what per H3 cell, mean per resolution-$themes_display cell, first three principal components as red, green and blue](assets/map_pca$themes_suffix.png)

$themes_legend

Each axis on its own, blue below zero and orange above. The $themes_what at both ends of each axis are in [`$themes_report`]($code_url/blob/master/docs/$themes_report).

![First principal component of the $themes_what per resolution-$themes_display cell](assets/map_pc1$themes_suffix.png)

![Second principal component of the $themes_what per resolution-$themes_display cell](assets/map_pc2$themes_suffix.png)

![Third principal component of the $themes_what per resolution-$themes_display cell](assets/map_pc3$themes_suffix.png)
<!-- files -->

$themes_files$themes_what, one vector each (see [Methodology](#methodology))
<!-- configs -->
, `$themes_folder`
<!-- method -->
$themes_step. **Embeddings of the $themes_what.** The $themes_texts $themes_selection embedded with [`$themes_model`](https://huggingface.co/$themes_model) (sentence-transformers, bf16, instruction `$themes_prompt`, last-token pooling, L2-normalised), stored as float16 in `$themes_folder` with their `id`, `lat` and `lon`; join on `id` for the rest. For the maps, each counts $themes_placement (resolution $themes_resolution), each cell takes the mean of its texts, each parent cell of resolution $themes_display the mean of its child cells (each child counts once), parents with fewer than $themes_min $themes_unit are dropped, and PCA is fitted on the parents' means (each counts once). Maps use the Equal Earth projection.
