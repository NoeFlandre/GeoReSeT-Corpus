#!/usr/bin/env bash
# CPU job: plan every unit of the release (GlotLID on website sentences, routing, deduplication).
#   oarsub -n georeset-plan -p "gpu_count=0" -l nodes=1,walltime=6:00 "bash ~/src/GeoReSeT-Corpus/scripts/g5k/plan_job.sh"
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
"$UV" run --project "$SRC" --frozen --no-sync georeset translate-plan --scratch "$WORK/scratch" --workers "${WORKERS:-$(( $(nproc) / 4 ))}"
