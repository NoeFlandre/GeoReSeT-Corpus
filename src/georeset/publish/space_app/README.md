---
title: GeoReSeT Corpus Fused
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: other
short_description: Explore where the GeoReSeT corpus speaks about the land
datasets:
  - NoeFlandre/GeoReSeT-Corpus-Fused
---

# GeoReSeT Corpus Fused: explorer

An interactive map of [GeoReSeT Corpus Fused](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus-Fused):
sentences tied to OpenStreetMap polygons and scene descriptions of geotagged photos, on H3 cells.

Only the sentences labelled relevant to land use are shown, from every source or one (websites, OSM
descriptions, Wikipedia, Wikivoyage). Every number is counted per H3 cell of resolution 5; zoomed out, each
coarser cell shows the mean of its resolution-5 cells, so every place counts once. Layers: land-use
sentences, scene photos, the most frequent language, and the dataset card's PCA maps of what places talk
about. Click a cell for its numbers, sample sentences and a scene description.

The files in `data/` are built by `georeset space` from one revision of the dataset, named in
`data/manifest.json` and in the app; the app says when the dataset has changed since.
