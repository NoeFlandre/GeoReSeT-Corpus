#!/usr/bin/env bash
# GPU job: label the planned texts with the benchmark's model (SGLang + DSpark) on this node's GPU.
# Usage: LABEL_BUCKET=<owner>/<bucket> label_job.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
# SGLang's kernels need the CUDA toolkit (as in the benchmark's node job).
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load "${LABEL_CUDA_MODULE:-cuda-toolkit/12.9.1}" 2>/dev/null || module load cuda-toolkit 2>/dev/null || true
command -v nvcc >/dev/null || { echo "no CUDA toolkit (nvcc) on this node" >&2; exit 6; }
CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_HOME LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
# LFM2.5 declares 131072 context tokens; SGLang's derived default is 128000 (as in the benchmark).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 FLASHINFER_WORKSPACE_BASE="$WORK/flashinfer"
# outlines-core (a dependency of SGLang 0.5.20) has no wheel on these nodes and builds from source: needs Rust.
if ! command -v cargo >/dev/null; then
  export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  export PATH="$CARGO_HOME/bin:$PATH"
fi
"$UV" sync --project "$SRC" --frozen --no-dev --extra label --quiet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$UV" run --project "$SRC" --frozen --no-sync georeset label-work \
  --bucket "${LABEL_BUCKET:?set LABEL_BUCKET}" --scratch "$WORK/scratch"
