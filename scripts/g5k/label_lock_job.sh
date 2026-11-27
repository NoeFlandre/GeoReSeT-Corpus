#!/usr/bin/env bash
# CPU job: refresh uv.lock for the label and tokenize extras (resolution runs on Grid'5000, not on a laptop).
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" lock --project "$SRC"
