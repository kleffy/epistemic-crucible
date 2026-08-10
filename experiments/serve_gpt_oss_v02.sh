#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${repo_dir}/.venv/bin:${PATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

cd "${repo_dir}"
exec .venv/bin/vllm serve openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --tokenizer-revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --quantization mxfp4 \
  --load-format safetensors \
  --max-model-len 8192 \
  --reasoning-parser openai_gptoss \
  --served-model-name openai/gpt-oss-20b \
  --seed 0 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --no-enable-chunked-prefill \
  --host 127.0.0.1 \
  --port 8000
