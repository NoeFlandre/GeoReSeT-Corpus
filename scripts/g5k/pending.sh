#!/usr/bin/env bash
# g5k-dispatch's `pending` command: prints the work items. Its environment is kept in the dispatcher
# job's scratch folder ($DISPATCH_SCRATCH), deleted with that job.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="${DISPATCH_SCRATCH:?run by g5k-dispatch}/georeset"
export UV_CACHE_DIR="$ROOT/uv-cache" UV_PROJECT_ENVIRONMENT="$ROOT/venv" HF_HOME="$ROOT/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
UV="${DISPATCH_UV:?run by g5k-dispatch}"
"$UV" run --project "$SRC" --frozen --no-dev --quiet georeset "${1:-translate-pending}"
