#!/usr/bin/env bash
# CPU job: compare the labelling check's decisions with the dataset's labels; with LABEL_REMOVE=1, delete its bucket.
# Usage: LABEL_BUCKET=<owner>/<bucket> [LABEL_REMOVE=1] label_report_job.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset label-report \
  --bucket "${LABEL_BUCKET:?set LABEL_BUCKET}" --scratch "$WORK/scratch"
if [[ "${LABEL_REMOVE:-0}" == 1 ]]; then
  "$UV" run --project "$SRC" --frozen --no-sync georeset label-remove --bucket "$LABEL_BUCKET"
fi
