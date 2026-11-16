#!/usr/bin/env bash
# CPU job, once every shard is embedded: add embeddings_en to a copy of the release and write the card.
# It ends by opening a pull request on the dataset; nothing reaches main until embed-finalize.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$WORK/release"
"$UV" sync --project "$SRC" --frozen --no-dev --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset embed-assemble --work "$OUT/work" --out "$OUT/out"
