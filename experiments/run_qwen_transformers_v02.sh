#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"

exec .venv/bin/python experiments/run_factorial_eval.py "$@"
