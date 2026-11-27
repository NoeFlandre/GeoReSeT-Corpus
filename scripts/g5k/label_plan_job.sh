#!/usr/bin/env bash
# CPU job: sample corpus rows and upload their tokenized prompts to the labelling check's bucket.
# Usage: LABEL_BUCKET=<owner>/<bucket> [LABEL_SAMPLE=48] label_plan_job.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra tokenize --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset label-plan \
  --bucket "${LABEL_BUCKET:?set LABEL_BUCKET}" --scratch "$WORK/scratch" --sample "${LABEL_SAMPLE:-48}"
