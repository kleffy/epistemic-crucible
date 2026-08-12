#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
REVISION="95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
PROJECT_VENV_BIN="${PWD}/.venv/bin"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export VLLM_BATCH_INVARIANT="1"
# FlashInfer invokes Ninja as a subprocess during kernel warm-up.
export PATH="${PROJECT_VENV_BIN}:${PATH}"

exec .venv/bin/vllm serve "$MODEL_ID" \
  --revision "$REVISION" \
  --tokenizer-revision "$REVISION" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --tokenizer-mode mistral \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --ignore-patterns consolidated.safetensors \
  --generation-config vllm \
  --language-model-only \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --seed 0 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --enable-chunked-prefill
