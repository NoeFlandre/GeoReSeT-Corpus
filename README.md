# GeoReSeT Corpus

Builds and publishes [`NoeFlandre/GeoReSeT-Corpus`](https://huggingface.co/datasets/NoeFlandre/GeoReSeT-Corpus):
multilingual sentences tied to OpenStreetMap polygons, each labelled for land-use / land-cover relevance.

```bash
uv sync
uv run georeset run --publish   # end to end, resumable
make check                      # lint, types, tests (100% coverage), CRAP < 6
make docs                       # documentation site
```

See [`docs/`](docs/index.md) for the pipeline and commands.
