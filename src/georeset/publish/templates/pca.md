## Principal components

Each sentence is turned into a list of 1,024 numbers, its embedding, so that sentences about similar things get similar lists. The maps show, place by place, which kinds of text are common, along three main directions, shown as red, green and blue.

### How it is computed

1. **Embed.** Each sentence becomes an embedding with [`$model`]($model_url) (multilingual maps: $texts sentences).
2. **Average per place.** Sentences are grouped by [H3](https://h3geo.org) cell, about 1,800 km² each. Each cell takes the average of its sentences' embeddings, so every place counts once, however much is written there. Places with fewer than $min_texts sentences are left out.
3. **Find the directions.** [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) finds the three directions along which these place averages differ the most. Each direction is an axis, and each place gets a score on each axis.
4. **Colour.** A place's scores on axes 1, 2 and 3 become its red, green and blue, scaled between each axis' 2nd and 98th percentile. Places with similar colours talk about similar things.
5. **Name the axes.** Each axis has two ends: the sentences scoring highest and lowest on it. The names below come from reading them. All of them are listed in the axes reports ([multilingual]($report_landuse), [English]($report_pool)).

**Variance** is the share of all the differences between places that an axis captures.

**Language explains** is the share of an axis' differences that the language of the sentences explains. Above 50%, the axis mostly separates languages rather than topics.

### Multilingual: all relevant sentences

Each place's sentences, in their original languages ($hexagons places).

$landuse_block

### English: original and translated

The relevant sentences written in English, plus the [English translations](#english-translations) of the others, pooled ($pool_texts sentences, $pool_hexagons places). Their embeddings come from the same model, so both kinds sit in one space. Each sentence keeps the language it was written in, so the language share shows how much the source language still shapes the English view.

$pool_block

