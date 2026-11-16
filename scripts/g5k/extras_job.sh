#!/usr/bin/env bash
# CPU job: copy the fused dataset's embeddings and EUNIS label shares into the latest release, check them
# and write the card. It ends by opening a pull request on the dataset; nothing reaches main until
# extras-finalize.
set -euo pipefail
source "$(dirname "$0")/env.sh"
df -h /tmp | tail -1
OUT="$WORK/release"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset extras-propose --work "$OUT/work" --out "$OUT/out"
