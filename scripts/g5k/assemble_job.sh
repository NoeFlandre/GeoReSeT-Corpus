#!/usr/bin/env bash
# CPU job, once every shard is translated: add the columns to a copy of the release and write the card.
# It ends by opening a pull request on the dataset; nothing reaches main until translate-finalize.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$WORK/release"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset translate-assemble \
  --work "$OUT/work" --out "$OUT/out" --scratch "$WORK/scratch"
