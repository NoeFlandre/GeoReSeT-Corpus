#!/usr/bin/env bash
# GPU job (submitted by g5k-dispatch): translate work shards for about 50 minutes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
# CTranslate2 loads cuBLAS and cuDNN from the nvidia-* wheels.
LD_LIBRARY_PATH="$(find "$WORK/venv" -path '*/nvidia/*/lib' -type d | paste -sd:):${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$UV" run --project "$SRC" --frozen --no-sync georeset translate-work \
  --scratch "$WORK/scratch" --offset "${OAR_JOB_ID:-0}" --minutes "${MINUTES:-50}"
