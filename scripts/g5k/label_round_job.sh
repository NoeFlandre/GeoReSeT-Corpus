#!/usr/bin/env bash
# CPU job: build round 1 (labels joined to the filter's generations, data/ without its inline landuse) and
# open the corpus pull request with it. Runs on Grid'5000; the folder lives on node-local scratch.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra tokenize --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset label-round-build \
  --work "$WORK/round" --out "$WORK/round-out"
"$UV" run --project "$SRC" --frozen --no-sync georeset label-round-propose --out "$WORK/round-out"
