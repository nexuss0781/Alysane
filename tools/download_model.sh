#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REV=1d461723eec654e65efdc40cf49301c89c0c92f4
BASE=https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/$REV
mkdir -p models
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  if [ ! -f "models/$f" ]; then
    curl -L --fail --retry 2 -o "models/$f" "$BASE/$f"
  fi
done
sha256sum models/model.safetensors
