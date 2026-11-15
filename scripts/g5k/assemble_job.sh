#!/usr/bin/env bash
# CPU job, once every shard is translated: add the columns to a copy of the release and write the card.
# Publishing is a separate, manual step (translate-publish) on the output this job leaves in $OUT.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="${OUT:?set OUT to a node-local folder kept after the job, e.g. /tmp/georeset-release}"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset translate-assemble \
  --work "$OUT/work" --out "$OUT/out" --scratch "$WORK/scratch"
