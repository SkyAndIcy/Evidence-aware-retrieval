#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  generator)
    model='Qwen/Qwen3-8B'
    served='Qwen3-8B'
    revision="${EARA_GENERATOR_REVISION:?Set an immutable checkpoint revision first}"
    port="${EARA_GENERATOR_PORT:-8000}"
    ;;
  reviewer)
    model='Qwen/Qwen3-4B'
    served='Qwen3-4B'
    revision="${EARA_SMALL_REVISION:?Set an immutable checkpoint revision first}"
    port="${EARA_SMALL_PORT:-8001}"
    ;;
  *)
    printf '%s\n' 'Usage: bash experiments/serve_qwen.sh generator|reviewer' >&2
    exit 2
    ;;
esac
command -v vllm >/dev/null || { printf '%s\n' 'Install and validate vLLM in a separate model-serving environment first.' >&2; exit 2; }
exec vllm serve "$model" --revision "$revision" --served-model-name "$served" --host 127.0.0.1 --port "$port" --max-model-len 32768 --generation-config vllm
