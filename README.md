# GeoReSeT Corpus

Builds and publishes [`NoeFlandre/GeoReSeT-Corpus`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus):
multilingual sentences tied to OpenStreetMap polygons, each labelled for land-use / land-cover relevance,
and [`NoeFlandre/GeoReSeT-Corpus-Fused`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus-Fused):
the same corpus plus scene descriptions of geotagged photos from [geo-rag](https://github.com/ananthu-aniraj/geo-rag).

```bash
uv sync
uv run georeset run --publish   # end to end, resumable
uv run georeset fuse --publish  # corpus release v1.0 + geo-rag scenes -> GeoReSeT-Corpus-Fused
make check                      # lint, types, tests (100% coverage), CRAP < 6
make docs                       # documentation site
```

See [`docs/`](docs/index.md) for the pipeline and commands.
