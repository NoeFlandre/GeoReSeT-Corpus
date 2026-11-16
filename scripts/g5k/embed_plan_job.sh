#!/usr/bin/env bash
# CPU job, once: upload the English texts to embed to the bucket as work shards.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset embed-plan --scratch "$WORK/scratch"
