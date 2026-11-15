#!/usr/bin/env bash
# CPU job, once: convert NLLB-200 3.3B (the benchmark's revision) to CTranslate2 float16, upload it to the bucket.
#   oarsub -n georeset-convert -p "gpu_count=0" -l nodes=1,walltime=1:00 "bash ~/src/GeoReSeT-Corpus/scripts/g5k/convert_model.sh"
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$UV" sync --project "$SRC" --frozen --no-dev --extra translate --quiet
"$UV" run --project "$SRC" --frozen --no-sync --with torch ct2-transformers-converter \
  --model facebook/nllb-200-3.3B --revision 1a07f7d195896b2114afcb79b7b57ab512e7b43e \
  --output_dir "$WORK/model" --quantization float16 \
  --copy_files tokenizer.json sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json
"$UV" run --project "$SRC" --frozen --no-sync georeset translate-upload-model "$WORK/model"
