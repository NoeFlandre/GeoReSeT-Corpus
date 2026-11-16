#!/usr/bin/env bash
# CPU job, after the axis names are reviewed: the corpus card with the two PCA maps, and a pull request.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$WORK/release"
"$UV" sync --project "$SRC" --frozen --no-dev --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset pca-propose --out "$OUT/out" --work "$OUT/work"
