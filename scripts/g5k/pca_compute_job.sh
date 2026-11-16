#!/usr/bin/env bash
# CPU job: PCA of the multilingual and English (original and translated) relevant sentences of release v1.3;
# the outputs go to the bucket for pca_propose_job.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
df -h /tmp | tail -1
"$UV" sync --project "$SRC" --frozen --no-dev --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset pca-compute --scratch "$WORK/scratch" --work "$WORK/work"
