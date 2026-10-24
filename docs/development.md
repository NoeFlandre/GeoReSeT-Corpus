# Development

```bash
make check    # ruff format + lint, ty, pytest with 100% branch coverage, CRAP < 6
make mutate   # mutation testing with mutmut
make docs     # this site
```

* **Tests** are written first and run on small synthetic tables shaped like the upstream files; no network.
* **Mutation testing** (`make mutate`, mutmut in fork-server mode because polars' thread pool does not
  survive `fork()`): every mutant is killed except equivalent ones, which are marked
  `# pragma: no mutate (<reason>)` (speed-only settings, idempotent setup, scratch names).
  `publish/figures.py` holds only matplotlib styling and is excluded; its maths lives in `publish/geo.py`.
* **CRAP** (`tools/crap.py`) combines radon complexity and per-function coverage: `cc² × (1 − cov)³ + cc`.
  Every function must stay under 6.
* **Speed**: polars throughout, list columns exploded before joins, streaming scans for large text columns,
  parallel downloads with a cap on concurrent conversions.
