# Sourced by the job scripts: everything a job writes lives in node-local /tmp/georeset-<job id> and is
# deleted when the job ends. uv itself is installed there; the Hugging Face token is read from $HOME.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="/tmp/georeset-${OAR_JOB_ID:-$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export UV_CACHE_DIR="$WORK/uv-cache" UV_PROJECT_ENVIRONMENT="$WORK/venv" UV_PYTHON_INSTALL_DIR="$WORK/python"
export HF_HOME="$WORK/hf" XDG_CACHE_HOME="$WORK/xdg" PIP_CACHE_DIR="$WORK/pip"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
python3 -m venv "$WORK/uv" && "$WORK/uv/bin/pip" install --quiet uv
UV="$WORK/uv/bin/uv"
echo "node: $(hostname)  job: ${OAR_JOB_ID:-none}  code: $(git -C "$SRC" rev-parse --short HEAD)  start: $(date)"
