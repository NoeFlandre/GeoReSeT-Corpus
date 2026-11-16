#!/usr/bin/env bash
# GPU job (submitted by g5k-dispatch): embed work shards for about 50 minutes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra embed --quiet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$UV" run --project "$SRC" --frozen --no-sync georeset embed-work \
  --scratch "$WORK/scratch" --offset "${OAR_JOB_ID:-0}" --minutes "${MINUTES:-50}"
