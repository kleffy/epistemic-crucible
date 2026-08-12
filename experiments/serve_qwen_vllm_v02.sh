#!/usr/bin/env bash
set -euo pipefail

# Archived failed preflight profile. vLLM 0.26.0 rejects batch-invariant mode
# for Qwen3.6's GDN_ATTN before the API server starts; no evaluation trace was
# produced. The selected pilot backend is pinned direct Transformers instead.
MODEL_ID="Qwen/Qwen3.6-27B"
REVISION="ea24fab73f0d8f18a59b05d79f5d2312a859b21a"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export VLLM_BATCH_INVARIANT="1"

exec .venv/bin/vllm serve "$MODEL_ID" \
  --revision "$REVISION" \
  --tokenizer-revision "$REVISION" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --language-model-only \
  --reasoning-parser qwen3 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --seed 0 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --enable-chunked-prefill
